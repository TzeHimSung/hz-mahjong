# 杭州麻将 AI

[English](README.md) | 简体中文

【杭州麻将对战平台】的 Python 客户端，采用固定的**单步期望策略**，提供三种对战模式和可恢复的本地运行记录。

## 快速上手

### 1. 安装

需要 Python **3.12+**，以下命令均在仓库根目录执行。安装时会构建 C 扩展：Windows AMD64 使用项目声明的 Zig 构建依赖；Linux 需要 C 编译器、匹配版本的 Python 开发头文件，以及 Python 的 venv 支持。

**Windows — PowerShell**

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --editable .\heuristic_ai
if (-not (Test-Path -LiteralPath '.\heuristic_ai\config.yaml')) {
    Copy-Item -LiteralPath '.\heuristic_ai\config.example.yaml' -Destination '.\heuristic_ai\config.yaml'
}
```

**Linux — Bash**

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install --editable ./heuristic_ai
if [ ! -e ./heuristic_ai/config.yaml ]; then
    cp ./heuristic_ai/config.example.yaml ./heuristic_ai/config.yaml
fi
```

上述命令保留已有配置。安装和运行使用同一个虚拟环境；虚拟环境与原生扩展应在目标操作系统上构建。

### 2. 配置模式

编辑 `heuristic_ai/config.yaml`，以 [config.example.yaml](heuristic_ai/config.example.yaml) 为完整模板，将 `REPLACE_ME_...` 替换为自己的平台连接信息、房间标识和凭据。本地配置已被 Git 忽略。

保留 `schema_version: 3`。**无论选择哪种模式，`train` 和 `race` 都必须完整填写。** 所有提供的区块都会校验，包含存在时的 `match`，但只运行选中的模式。模板默认选择 `race`。

| 模式 | 用途与身份要求 | 结束方式 |
|---|---|---|
| `train` | 四个不同身份进入同一个已有测试房；填写房间 ID，以及四组不同的名称和令牌。这里指测试房对战，不是模型训练。 | 执行一轮完整对战后退出。启动时房间须为 `registering` 或 `finished`；客户端不建房，也不循环重开。 |
| `race` | 已完成报名的正式赛身份；填写赛事房间 ID 和参赛令牌。 | 协调 ready、阶段和分配的对局，直到权威终态、淘汰或用户中断。首次报名在平台完成。 |
| `match` | 已有全局身份的令牌，填入顶层 `global_token`；不配置匹配房间 ID。 | 默认完成一个会话后退出；`match.continuous: true` 时，当前会话结束并成功发布记录后继续下一次匹配。 |

使用 `match` 时，在复制出的完整模板中修改以下**顶层字段**，保留其余区块：

```yaml
mode: match
global_token: REPLACE_ME_GLOBAL_TOKEN
```

所有保留区块的 `server.url` 都需要填写。`server.insecure: true` 表示跳过 TLS 证书校验，模板目前使用此值。相对的 `competition.output_dir` 以配置文件所在目录为基准；模板中的 `../competition_record` 因此指向仓库根目录下的 `competition_record/`。

### 3. 启动与停止

填写完整配置后，选择一种启动命令：

**Windows — PowerShell**

```powershell
& .\.venv\Scripts\python.exe -B .\heuristic_ai\launch.py
```

**Linux — Bash**

```bash
./.venv/bin/python -B ./heuristic_ai/launch.py
```

[launch.py](heuristic_ai/launch.py) 固定读取它旁边的 `config.yaml`，不依赖当前工作目录。配置占位符会在联网前被拒绝。运行模式由 YAML 决定，没有 `--mode` 参数。

若使用另一份完整配置，将脚本入口换成模块入口：

- Windows：`& .\.venv\Scripts\python.exe -B -m heuristic_ai.runtime.runner --config "REPLACE_ME_CONFIG_PATH"`
- Linux：`./.venv/bin/python -B -m heuristic_ai.runtime.runner --config "REPLACE_ME_CONFIG_PATH"`

进度写入 **stderr**。按 `Ctrl+C` 停止前台运行，让程序收束工作进程和记录。切换模式前先停止当前进程；停止客户端不等于关闭服务端房间。

## 技术说明

### 代码结构

| 模块 | 职责 |
|---|---|
| [config](heuristic_ai/src/heuristic_ai/config/) | 严格加载 YAML，检查 schema，构造类型化配置并拒绝占位符。 |
| [protocol](heuristic_ai/src/heuristic_ai/protocol/) | 认证请求头、API 请求、HTTP 传输、重试策略和快照归一化。 |
| [runtime](heuristic_ai/src/heuristic_ai/runtime/) | 模式协调、对局工作单元、I/O 进程、动作截止、守护、恢复和进度。 |
| [game](heuristic_ai/src/heuristic_ai/game/) | 牌码、观察状态、合法动作、计分、手牌求解与公开库存。 |
| [strategy](heuristic_ai/src/heuristic_ai/strategy/) | 单步期望搜索、静态估值、预计算、缓存和原生计算内核。 |
| [artifacts](heuristic_ai/src/heuristic_ai/artifacts/) | 决策记录、工作分片、校验导出、回放解析和恢复清理。 |

### 状态、决策与提交

三种对战模式共用游戏规则与决策控制器：

```text
SSE 通知 / 状态轮询
  -> 校验后的快照与公开库存
  -> 合法动作候选
  -> 单步期望策略决策或降级结果
  -> 动作提交与确认
  -> 决策记录与运行归档
```

[io_process.py](heuristic_ai/src/heuristic_ai/runtime/io_process.py) 将网络工作放在独立进程中，HTTP 客户端及其异步事件循环由该进程持有，与策略计算分开。在 [httpx_transport.py](heuristic_ai/src/heuristic_ai/protocol/httpx_transport.py) 中，通知连接允许协商 HTTP/2，state、action 和 control 通道使用 HTTP/1.1。

SSE 通知提供序号提示，不提供权威牌面或动作窗口时间。客户端取得状态快照后再生成动作。公开库存按可见实体牌核算，包含被认领弃牌的转移，避免把同一张牌重复计数。

安全读取采用有界重试。动作提交丢失回执后不会自动重放；HTTP 请求成功与目标动作得到执行确认是不同事实。恢复过程会保留已确认或结果未知的提交状态，以及原动作预算。

### 单步期望策略

策略英文名称为 **One-Ply Expectation Policy**，代码标识为 [`one_ply_expectation`](heuristic_ai/src/heuristic_ai/strategy/models.py)。不提供策略选择配置：搜索上限固定为 **20,000 个逻辑节点**，并启用缓存。

[one_ply_search.py](heuristic_ai/src/heuristic_ai/strategy/search/one_ply_search.py) 对每个合法候选及认领后的必弃后继估值，再投影到下一次本人摸牌。每种可能摸牌以 `max(0, 4 - known_count)` 加权；叶子有合法胡牌时选择胡牌，否则比较弃牌。加权估值向量按字典序比较，同值时由静态估值决胜。

向量包含归一化胡牌收益、非终局进展、活牌权重、番数潜力、动作链潜力和路线数。这些是局部决策模型的组成部分，不是完整四人对局最终成绩的预测。

模型只使用可获得的公开信息和本人手牌。期间对手摸牌会计入牌墙消耗，但不模拟对手的隐藏手牌或动作。未见实体牌也可能位于对手手中，其权重不能解释为已知牌墙分布。

预计算在预测状态上运行同一个单步期望模型，身份、恢复代次和输入核对通过后才复用结果。[C 扩展](heuristic_ai/src/heuristic_ai/strategy/native/_one_ply_native.c) 提供叶子估值和 JSON 辅助计算；扩展不可用时，运行代码可使用 Python 实现。

| 执行来源 | 含义 |
|---|---|
| `completed_search` | 完成全部根候选的比较。 |
| `prefetch_hit` | 复用已校验的完整预计算决策。 |
| `forced_action` | 当前只有一个合法候选。 |
| `static_fallback` | 搜索无法交付可用结果时采用静态估值。 |
| `legal_fallback` | 无法采用更完整估值时，使用确定性的合法动作保底。 |

公开库存未知、历史证据不足、预算耗尽或计算失败，都可能产生带原因记录的降级。降级是单步期望策略的内部执行路径，不是另一种可选择的策略。

### 预算与恢复

| 配置或约束 | 当前模板／默认值 | 含义 |
|---|---|---|
| `competition.decision_budget_ms` | `train`：3000；`race`：2000；`match`：3000 | 客户端决策预算；实际动作窗口可能进一步压缩它。 |
| 动作提交预留 | 300 ms | 弃牌和响应提交的内部默认值，不是 YAML 字段。 |
| `match.matchmaking.max_parallel_games` | 10 | 可配置的客户端容量，校验范围为 1–16。 |
| `match.matchmaking.max_rounds_per_game` | 8 | 发起匹配时上报的配置上限；实际房间规模由平台返回。 |
| `race.supervision` | 默认开启；心跳超时 30 s，停滞超时 180 s | 前台守护设置，不是服务端动作超时。 |

搜索同时遵守节点上限和剩余动作时间。重复快照、重试和工作单元恢复，不会为同一次动作机会重新发放完整的长搜索预算。预算和桌数上限属于客户端设置；房间规则及对局结果以平台响应为准。

单局恢复按游戏和恢复代次跟踪：停止受影响的工作单元，保存提交状态及恢复证据，退避后协调下一代工作单元。正式赛另有前台父进程负责心跳与停滞检查。Windows 命令行入口限制全机同时运行一个 `race` 实例。

### 记录与离线恢复

在配置的输出根目录下，主要路径如下：

| 路径 | 内容 |
|---|---|
| `.record_work/<mode>/<run-id>/` | 录制和恢复使用的受管工作分片。 |
| `test_records/<run-id>/` | `train` 和 `match` 的默认导出目录。 |
| `official_records/<run-id>/` | `race` 导出目录；`train` 设置 `save_to_official: true` 时也写入这里。 |
| `runtime_logs/race/<supervision-id>/` | 守护进程日志与运行关联信息。 |

发布后的记录包含 `manifest.json` 和 `decisions.jsonl`，恢复事件另写入 `recovery.jsonl`。train 还导出分局回放与状态轮询记录；race 和 match 不包含同样的四家完整回放数据。

运行状态（`completed`、`degraded`、`failed`、`interrupted`）与导出是否完整分别记录。序列化边界会脱敏凭据和敏感认证字段，但记录仍可能含有玩家名称、用户标识和对局标识。

[artifacts.cli](heuristic_ai/src/heuristic_ai/artifacts/cli.py) 提供两种独立的离线操作。先停止运行，再打开其工作分片进行恢复。将路径占位符替换为受管工作目录；显式指定输出时，使用新的目标目录。

**合并导出 — Windows**

```powershell
& .\.venv\Scripts\python.exe -B -m heuristic_ai.artifacts.cli merge --work-dir "REPLACE_ME_WORK_DIR" --output-dir "REPLACE_ME_OUTPUT_DIR"
```

**合并导出 — Linux**

```bash
./.venv/bin/python -B -m heuristic_ai.artifacts.cli merge --work-dir "REPLACE_ME_WORK_DIR" --output-dir "REPLACE_ME_OUTPUT_DIR"
```

`merge` 校验保留的分片并写出新的归档，`--output-dir` 可省略。导出后，符合条件的 completed/degraded 运行可能自动清理工作材料；failed/interrupted 运行保留恢复材料。

**重试清理 — Windows**

```powershell
& .\.venv\Scripts\python.exe -B -m heuristic_ai.artifacts.cli cleanup --work-dir "REPLACE_ME_WORK_DIR"
```

**重试清理 — Linux**

```bash
./.venv/bin/python -B -m heuristic_ai.artifacts.cli cleanup --work-dir "REPLACE_ME_WORK_DIR"
```

`cleanup` 核对已发布归档及清理凭据后才重试删除，用于清理尚未完成的情况，不是每次 merge 后都要执行的步骤。两种命令均输出 JSON，只操作本地文件，不读取平台凭据，也不启动联网对战。
