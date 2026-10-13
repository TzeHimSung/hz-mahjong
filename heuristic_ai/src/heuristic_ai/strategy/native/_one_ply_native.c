/* Exact, witness-free One-Ply Expectation leaf evaluator.
 *
 * The dynamic program is independently checked against one_ply_solver.py and
 * the shared rules/scoring implementations. Fixed-size direct-mapped caches
 * affect work only, never values. All entrypoints keep the GIL, so cache clears
 * and concurrent callers cannot race. No worker/thread is created here.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define COMP_CAP (1u << 18)
#define PROFILE_CAP (1u << 14)
#define PAIR_CAP (1u << 15)
#define SUMMARY_CAP (1u << 16)
#define LOCAL_MASK UINT64_C(511)
#define PAIR_MASK UINT64_C(262143)
#define NATURAL_MASK ((UINT64_C(1) << 33) - 1)
#define ALL_MASK ((UINT64_C(1) << 34) - 1)

typedef struct { uint64_t key; uint32_t value, generation; } ComponentEntry;
typedef struct { uint64_t key; uint32_t values[10], generation; } ProfileEntry;
typedef struct { uint64_t key, values[10]; uint32_t generation; } PairEntry;
typedef struct {
    int regular_distance, pairs_distance, minimum, route_count, whites, quads;
    uint64_t regular_mask, pairs_mask, improving_mask;
} Summary;
typedef struct { uint64_t low, high; Summary value; uint32_t generation; } SummaryEntry;

static ComponentEntry component_cache[COMP_CAP];
static ProfileEntry profile_cache[PROFILE_CAP];
static PairEntry pair_cache[PAIR_CAP];
static SummaryEntry summary_cache[SUMMARY_CAP];
static uint32_t generation = 1;
static uint64_t hits = 0, misses = 0;

static uint64_t hash64(uint64_t value) {
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

static void clear_generation(void) {
    if (++generation == 0) {
        memset(component_cache, 0, sizeof(component_cache));
        memset(profile_cache, 0, sizeof(profile_cache));
        memset(pair_cache, 0, sizeof(pair_cache));
        memset(summary_cache, 0, sizeof(summary_cache));
        generation = 1;
    }
}

static int popcount(uint64_t value) {
    value -= (value >> 1) & UINT64_C(0x5555555555555555);
    value = (value & UINT64_C(0x3333333333333333)) + ((value >> 2) & UINT64_C(0x3333333333333333));
    value = (value + (value >> 4)) & UINT64_C(0x0f0f0f0f0f0f0f0f);
    return (int)((value * UINT64_C(0x0101010101010101)) >> 56);
}

static uint32_t prefer(uint32_t left, uint32_t right) {
    if ((right >> 9) > (left >> 9)) return right;
    if ((right >> 9) == (left >> 9)) return left | right;
    return left;
}

static uint32_t component(uint32_t code, int groups, int pairs, int suited) {
    uint64_t key = (uint64_t)code | ((uint64_t)groups << 27)
                 | ((uint64_t)pairs << 30) | ((uint64_t)suited << 31);
    ComponentEntry *entry = &component_cache[hash64(key) & (COMP_CAP - 1)];
    if (entry->generation == generation && entry->key == key) {
        hits++;
        return entry->value;
    }
    misses++;
    uint32_t best;
    if (groups == 0 && pairs == 0) best = 0;
    else if (code == 0) best = suited ? 511 : 63;
    else {
        int first = 0;
        while (((code >> (3 * first)) & 7) == 0) first++;
        uint32_t unit = 1u << (3 * first);
        int count = (code >> (3 * first)) & 7;
        best = component(code - unit, groups, pairs, suited);
        if (pairs) {
            int used = count < 2 ? count : 2;
            uint32_t candidate = component(code - used * unit, groups, pairs - 1, suited) + (used << 9);
            if (used < 2) candidate |= 1u << first;
            best = prefer(best, candidate);
        }
        if (groups) {
            int used = count < 3 ? count : 3;
            uint32_t candidate = component(code - used * unit, groups - 1, pairs, suited) + (used << 9);
            if (used < 3) candidate |= 1u << first;
            best = prefer(best, candidate);
            if (suited) {
                int lower = first > 2 ? first - 2 : 0;
                int upper = first < 6 ? first : 6;
                for (int start = lower; start <= upper; start++) {
                    uint32_t remainder = code, missing = 0;
                    used = 0;
                    for (int index = start; index < start + 3; index++) {
                        if ((code >> (3 * index)) & 7) {
                            remainder -= 1u << (3 * index);
                            used++;
                        } else missing |= 1u << index;
                    }
                    candidate = (component(remainder, groups - 1, pairs, suited) + (used << 9)) | missing;
                    best = prefer(best, candidate);
                }
            }
        }
    }
    /* Recursive collisions may replace this slot; the completed value wins. */
    entry->key = key;
    entry->value = best;
    entry->generation = generation;
    return best;
}

static void profile(uint32_t code, int suited, uint32_t output[10]) {
    uint64_t key = (uint64_t)code | ((uint64_t)suited << 27);
    ProfileEntry *entry = &profile_cache[hash64(key) & (PROFILE_CAP - 1)];
    if (entry->generation == generation && entry->key == key) {
        hits++;
        memcpy(output, entry->values, sizeof(entry->values));
        return;
    }
    misses++;
    for (int groups = 0; groups < 5; groups++)
        for (int pairs = 0; pairs < 2; pairs++)
            output[2 * groups + pairs] = component(code, groups, pairs, suited);
    entry->key = key;
    memcpy(entry->values, output, sizeof(entry->values));
    entry->generation = generation;
}

static void paired_profile(uint32_t left_code, uint32_t right_code, int suited, uint64_t output[10]) {
    uint64_t key = (uint64_t)left_code | ((uint64_t)right_code << 27) | ((uint64_t)suited << 54);
    PairEntry *entry = &pair_cache[hash64(key) & (PAIR_CAP - 1)];
    if (entry->generation == generation && entry->key == key) {
        hits++;
        memcpy(output, entry->values, sizeof(entry->values));
        return;
    }
    misses++;
    uint32_t left[10], right[10];
    profile(left_code, 1, left);
    profile(right_code, suited, right);
    for (int groups = 0; groups < 5; groups++) {
        for (int pairs = 0; pairs < 2; pairs++) {
            int best_reuse = -1;
            uint64_t best_mask = 0;
            for (int lg = 0; lg <= groups; lg++) {
                for (int lp = 0; lp <= pairs; lp++) {
                    uint32_t first = left[2 * lg + lp];
                    uint32_t second = right[2 * (groups - lg) + pairs - lp];
                    int reuse = (first >> 9) + (second >> 9);
                    uint64_t mask = (first & LOCAL_MASK) | ((second & LOCAL_MASK) << 9);
                    if (reuse > best_reuse) { best_reuse = reuse; best_mask = mask; }
                    else if (reuse == best_reuse) best_mask |= mask;
                }
            }
            output[2 * groups + pairs] = ((uint64_t)best_reuse << 18) | best_mask;
        }
    }
    entry->key = key;
    memcpy(entry->values, output, sizeof(entry->values));
    entry->generation = generation;
}

static Summary summarize(const uint8_t counts[34], int melds) {
    uint64_t low = 0, high = (uint64_t)melds << 39;
    for (int index = 0; index < 21; index++) low |= (uint64_t)counts[index] << (3 * index);
    for (int index = 21; index < 34; index++) high |= (uint64_t)counts[index] << (3 * (index - 21));
    SummaryEntry *entry = &summary_cache[hash64(low ^ hash64(high)) & (SUMMARY_CAP - 1)];
    if (entry->generation == generation && entry->low == low && entry->high == high) {
        hits++;
        return entry->value;
    }
    misses++;
    uint32_t codes[4] = {0};
    for (int index = 0; index < 33; index++) codes[index / 9] |= (uint32_t)counts[index] << (3 * (index % 9));
    uint64_t left[10], right[10];
    paired_profile(codes[0], codes[1], 1, left);
    paired_profile(codes[2], codes[3], 0, right);
    int groups = 4 - melds, best_reuse = -1;
    uint64_t best_mask = 0;
    for (int lg = 0; lg <= groups; lg++) {
        for (int lp = 0; lp < 2; lp++) {
            uint64_t first = left[2 * lg + lp], second = right[2 * (groups - lg) + 1 - lp];
            int reuse = (first >> 18) + (second >> 18);
            uint64_t mask = (first & PAIR_MASK) | ((second & PAIR_MASK) << 18);
            if (reuse > best_reuse) { best_reuse = reuse; best_mask = mask; }
            else if (reuse == best_reuse) best_mask |= mask;
        }
    }
    Summary value;
    memset(&value, 0, sizeof(value));
    value.whites = counts[33];
    int natural_odds = 0;
    for (int index = 0; index < 33; index++) {
        if (counts[index] == 4) value.quads++;
        natural_odds += counts[index] & 1;
    }
    if (value.whites == 4 && natural_odds == 0) value.quads++;
    value.regular_distance = 3 * groups + 2 - best_reuse - value.whites;
    value.regular_mask = value.regular_distance ? (best_mask | (UINT64_C(1) << 33)) : 0;
    value.pairs_distance = -1;
    if (melds == 0) {
        int pairs = 0, odds = 0;
        uint64_t odd_mask = 0;
        for (int index = 0; index < 33; index++) {
            pairs += counts[index] / 2;
            if (counts[index] & 1) { odds++; odd_mask |= UINT64_C(1) << index; }
        }
        int remaining = 7 - pairs;
        int reused = 2 * pairs + (remaining < odds ? remaining : odds);
        value.pairs_distance = 14 - reused - value.whites;
        value.pairs_mask = value.pairs_distance ?
            ((remaining > odds ? NATURAL_MASK : odd_mask) | (UINT64_C(1) << 33)) : 0;
    }
    value.minimum = value.regular_distance;
    if (value.pairs_distance >= 0 && value.pairs_distance < value.minimum) value.minimum = value.pairs_distance;
    if (value.regular_distance == value.minimum) { value.improving_mask |= value.regular_mask; value.route_count++; }
    if (value.pairs_distance == value.minimum) { value.improving_mask |= value.pairs_mask; value.route_count++; }
    entry->low = low; entry->high = high; entry->value = value; entry->generation = generation;
    return value;
}

static PyObject *leaf(PyObject *self, PyObject *args) {
    const char *raw_counts, *raw_known;
    Py_ssize_t counts_size, known_size;
    int tile, melds, you_cai, baotou_before, chain, piao, catch_play, wall, factor, cache_enabled;
    if (!PyArg_ParseTuple(args, "y#y#iiiiiiiiii:leaf", &raw_counts, &counts_size, &raw_known, &known_size,
                          &tile, &melds, &you_cai, &baotou_before, &chain, &piao, &catch_play,
                          &wall, &factor, &cache_enabled)) return NULL;
    if (counts_size != 34 || known_size != 34 || tile < 0 || tile > 33 || melds < 0 || melds > 4
            || chain < 0 || chain > 6 || piao < 0 || piao > chain || wall < 20
            || (factor != 10 && factor != 24) || (you_cai != 0 && you_cai != 1)
            || (baotou_before != 0 && baotou_before != 1) || (catch_play != 0 && catch_play != 1)
            || (cache_enabled != 0 && cache_enabled != 1)) {
        PyErr_SetString(PyExc_ValueError, "invalid native leaf arguments"); return NULL;
    }
    uint8_t counts[34], known[34];
    memcpy(counts, raw_counts, 34); memcpy(known, raw_known, 34);
    int total = 0;
    for (int index = 0; index < 34; index++) {
        if (counts[index] > 4 || known[index] > 4 || known[index] < counts[index]) {
            PyErr_SetString(PyExc_ValueError, "invalid native leaf inventory"); return NULL;
        }
        total += counts[index];
    }
    if (total != 3 * (4 - melds) + 1 || known[tile] == 4 || counts[33] + piao + (tile == 33) > 4) {
        PyErr_SetString(PyExc_ValueError, "invalid native leaf hand or draw"); return NULL;
    }
    if (!cache_enabled) clear_generation();
    counts[tile]++; known[tile]++;
    Summary drawn = summarize(counts, melds);
    int baotou = baotou_before;
    int compatible = !you_cai || counts[33] == 0 || baotou || chain > 0;
    int winning = compatible && (baotou || drawn.regular_distance == 0 || drawn.pairs_distance == 0);
    int value[6] = {0}, selected = -1, cost = 1, terminal = 0, legal_count = 0;
    Summary selected_summary = drawn;
    if (winning) {
        int branch = drawn.regular_distance == 0 ? 1 : 0;
        if (drawn.pairs_distance == 0 && chain == piao) branch = 2 << drawn.quads;
        if (branch == 0) {
            if (!cache_enabled) clear_generation();
            PyErr_SetString(PyExc_ValueError, "native win has no compliant route"); return NULL;
        }
        int fan = branch * (1 << chain) * (counts[33] + piao == 4 ? 2 : 1) * (baotou ? 2 : 1);
        value[0] = factor * fan;
        terminal = 1;
    } else {
        uint64_t ones = 0, twos = 0, fours = 0;
        for (int index = 0; index < 34; index++) {
            uint64_t bit = UINT64_C(1) << index;
            if (known[index] == 4) fours |= bit;
            else { if (known[index] & 1) ones |= bit; if (known[index] & 2) twos |= bit; }
        }
        int best_distance = 99, best_live = -1, best_fan = -1, best_routes = -1;
        for (int index = 0; index < 34; index++) {
            if (!counts[index] || (catch_play && index != tile)
                    || (baotou && chain == 6 && index == 33)) continue;
            legal_count++;
            counts[index]--;
            Summary after = summarize(counts, melds);
            int local_compatible = !you_cai || after.whites == 0;
            int derived_baotou = after.minimum == 1 && after.improving_mask == ALL_MASK;
            int branch = 0;
            if (local_compatible || derived_baotou) {
                if (after.regular_distance == after.minimum) branch = 1;
                if (after.pairs_distance == after.minimum) branch = 2 << after.quads;
            }
            int fan = factor * branch * (after.whites == 4 ? 2 : 1) * (derived_baotou ? 2 : 1);
            uint64_t mask = after.improving_mask;
            int live = wall > 20 ? 4 * popcount(mask) - popcount(mask & ones)
                                  - 2 * popcount(mask & twos) - 4 * popcount(mask & fours) : 0;
            if (after.minimum < best_distance || (after.minimum == best_distance
                    && (live > best_live || (live == best_live && (fan > best_fan
                    || (fan == best_fan && after.route_count > best_routes)))))) {
                best_distance = after.minimum; best_live = live; best_fan = fan; best_routes = after.route_count;
                selected = index; selected_summary = after; compatible = local_compatible;
            }
            counts[index]++;
        }
        if (selected < 0) {
            if (!cache_enabled) clear_generation();
            PyErr_SetString(PyExc_ValueError, "native non-winning leaf has no legal discard"); return NULL;
        }
        cost += legal_count;
        if (wall < 24) terminal = 2;
        else { value[1] = 14 - best_distance; value[2] = best_live; value[3] = best_fan; value[4] = 1; value[5] = best_routes; }
    }
    if (!cache_enabled) clear_generation();
    return Py_BuildValue("((iiiiii)iiii(iKiKi))", value[0], value[1], value[2], value[3], value[4], value[5],
                         selected, cost, terminal, legal_count,
                         selected_summary.regular_distance, (unsigned long long)selected_summary.regular_mask,
                         selected_summary.pairs_distance, (unsigned long long)selected_summary.pairs_mask, compatible);
}

/* Exact internal static ordering fields for an already-discarded, one-short state.
 * Keep the independent Python evaluate_counts implementation as the oracle.
 */
static PyObject *evaluate_counts(PyObject *self, PyObject *args) {
    const char *raw_counts, *raw_known;
    Py_ssize_t counts_size, known_size;
    int melds, you_cai, baotou, chain, piao, draw_available, factor, cache_enabled;
    if (!PyArg_ParseTuple(args, "y#y#iiiiiiii:evaluate_counts",
                          &raw_counts, &counts_size, &raw_known, &known_size,
                          &melds, &you_cai, &baotou, &chain, &piao,
                          &draw_available, &factor, &cache_enabled)) return NULL;
    if (counts_size != 34 || known_size != 34 || melds < 0 || melds > 4
            || chain < 0 || chain > 6 || piao < 0 || piao > chain
            || (you_cai != 0 && you_cai != 1) || (baotou != 0 && baotou != 1)
            || (draw_available != 0 && draw_available != 1)
            || (factor != 10 && factor != 24) || (cache_enabled != 0 && cache_enabled != 1)) {
        PyErr_SetString(PyExc_ValueError, "invalid native evaluation arguments"); return NULL;
    }
    uint8_t counts[34], known[34];
    memcpy(counts, raw_counts, 34); memcpy(known, raw_known, 34);
    int total = 0;
    for (int index = 0; index < 34; index++) {
        if (counts[index] > 4 || known[index] > 4 || known[index] < counts[index]) {
            PyErr_SetString(PyExc_ValueError, "invalid native evaluation inventory"); return NULL;
        }
        total += counts[index];
    }
    if (total != 3 * (4 - melds) + 1 || counts[33] + piao > 4) {
        PyErr_SetString(PyExc_ValueError, "invalid native evaluation hand"); return NULL;
    }
    if (!cache_enabled) clear_generation();
    Summary value = summarize(counts, melds);
    int authoritative_baotou = baotou;
    int compatible = !you_cai || value.whites == 0 || authoritative_baotou || chain > 0;
    int derived_baotou = authoritative_baotou
        || (value.minimum == 1 && value.improving_mask == ALL_MASK);
    int branch = 0, live = 0;
    if (compatible || derived_baotou) {
        if (value.regular_distance == value.minimum) branch = 1;
        if (value.pairs_distance == value.minimum) branch = 2 << value.quads;
    }
    int fan = branch * factor * (value.whites + piao == 4 ? 2 : 1) * (derived_baotou ? 2 : 1);
    if (draw_available) {
        for (int index = 0; index < 34; index++)
            if (value.improving_mask & (UINT64_C(1) << index)) live += 4 - known[index];
    }
    if (!cache_enabled) clear_generation();
    return Py_BuildValue("((iiiiiii)(iKiKi))", 0, 0, -value.minimum, live, fan, 1 << chain, value.route_count,
                         value.regular_distance, (unsigned long long)value.regular_mask,
                         value.pairs_distance, (unsigned long long)value.pairs_mask, compatible);
}

static PyObject *clear_caches(PyObject *self, PyObject *args) {
    clear_generation(); hits = misses = 0;
    Py_RETURN_NONE;
}

static PyObject *cache_info(PyObject *self, PyObject *args) {
    unsigned long long bytes = sizeof(component_cache) + sizeof(profile_cache) + sizeof(pair_cache) + sizeof(summary_cache);
    return Py_BuildValue("{sK,sK,sK}", "hits", (unsigned long long)hits,
                         "misses", (unsigned long long)misses, "fixed_bytes", bytes);
}

/* Conservative owned-tree accounting without Python callbacks on every scalar.
 * Sized containers are private, acyclic search objects. Bounds deliberately
 * include spare capacity, GC headers and duplicate references. */
static size_t retained_bytes(PyObject *object, int depth) {
    if (depth > 100) { PyErr_SetString(PyExc_ValueError, "search object nesting is too deep"); return 0; }
    size_t size = _PySys_GetSizeOf(object);
    if (PyErr_Occurred()) return 0;
    if (PyTuple_CheckExact(object) || PyList_CheckExact(object)) {
        Py_ssize_t count = PyTuple_CheckExact(object) ? PyTuple_GET_SIZE(object) : PyList_GET_SIZE(object);
        for (Py_ssize_t index = 0; index < count; index++) {
            PyObject *item = PyTuple_CheckExact(object) ? PyTuple_GET_ITEM(object, index) : PyList_GET_ITEM(object, index);
            size += retained_bytes(item, depth + 1);
            if (PyErr_Occurred()) return 0;
        }
        return size;
    }
    if (PyDict_CheckExact(object)) {
        Py_ssize_t position = 0;
        PyObject *key, *value;
        while (PyDict_Next(object, &position, &key, &value)) {
            size += retained_bytes(key, depth + 1) + retained_bytes(value, depth + 1);
            if (PyErr_Occurred()) return 0;
        }
        return size;
    }
    if (PyUnicode_Check(object) || PyBytes_Check(object) || PyLong_Check(object)
            || object == Py_None || PyFloat_Check(object)) return size;
    /* GodState, Meld and NodeValue are tiny frozen records. Their referenced
     * tile strings/tuples fit within this allowance; a live frame is accounted
     * separately. No arbitrary user objects enter this internal interface. */
    return size + 1024;
}

static PyObject *retained_size(PyObject *self, PyObject *object) {
    size_t size = retained_bytes(object, 0);
    if (PyErr_Occurred()) return NULL;
    return PyLong_FromSize_t(size);
}

/* Exact built-in JSON trees are common in large search audits. Freeze those
 * without a Python frame for every scalar; unfamiliar subclasses/mappings
 * return NotImplemented to the existing Python validation path. */
static PyObject *freeze_tree(PyObject *object, int depth) {
    if (depth > 100) return Py_NewRef(Py_NotImplemented);
    if (object == Py_None || PyBool_Check(object)) return Py_NewRef(object);
    if (PyUnicode_CheckExact(object)) {
        if (!PyUnicode_AsUTF8(object)) {
            if (PyErr_ExceptionMatches(PyExc_UnicodeError)) {
                PyErr_Clear(); PyErr_SetString(PyExc_ValueError, "decision details must be JSON-serializable");
            }
            return NULL;
        }
        return Py_NewRef(object);
    }
    if (PyLong_CheckExact(object)) {
        /* All machine-sized integers serialize without the decimal digit
         * limit check. Keep Python's conversion/validation for larger values. */
        int overflow = 0;
        PyLong_AsLongLongAndOverflow(object, &overflow);
        if (PyErr_Occurred()) return NULL;
        if (!overflow) return Py_NewRef(object);
        PyObject *text = PyObject_Str(object);
        if (!text) {
            if (PyErr_ExceptionMatches(PyExc_ValueError) || PyErr_ExceptionMatches(PyExc_OverflowError)) {
                PyErr_Clear(); PyErr_SetString(PyExc_ValueError, "decision details must be JSON-serializable");
            }
            return NULL;
        }
        Py_DECREF(text);
        return Py_NewRef(object);
    }
    if (PyFloat_CheckExact(object)) {
        if (!isfinite(PyFloat_AS_DOUBLE(object))) {
            PyErr_SetString(PyExc_ValueError, "decision detail floats must be finite"); return NULL;
        }
        return Py_NewRef(object);
    }
    if (PyTuple_CheckExact(object) || PyList_CheckExact(object)) {
        int tuple = PyTuple_CheckExact(object);
        Py_ssize_t count = tuple ? PyTuple_GET_SIZE(object) : PyList_GET_SIZE(object);
        /* Reuse an exact tuple only after validating every descendant and
         * proving that none requires replacement. Lists, mappings and scalar
         * subclasses cannot satisfy this identity check. Allocate a copy at
         * the first changed child, keeping arbitrary input aliases isolated.
         */
        PyObject *result = tuple ? NULL : PyTuple_New(count);
        if (!tuple && !result) return NULL;
        for (Py_ssize_t index = 0; index < count; index++) {
            PyObject *original = tuple ? PyTuple_GET_ITEM(object, index) : PyList_GET_ITEM(object, index);
            PyObject *value = freeze_tree(original, depth + 1);
            if (!value || value == Py_NotImplemented) { Py_XDECREF(result); return value; }
            if (!result && value == original) {
                Py_DECREF(value);
                continue;
            }
            if (!result) {
                result = PyTuple_New(count);
                if (!result) { Py_DECREF(value); return NULL; }
                for (Py_ssize_t previous = 0; previous < index; previous++)
                    PyTuple_SET_ITEM(result, previous, Py_NewRef(PyTuple_GET_ITEM(object, previous)));
            }
            PyTuple_SET_ITEM(result, index, value);
        }
        return result ? result : Py_NewRef(object);
    }
    if (PyDict_CheckExact(object) || Py_TYPE(object) == &PyDictProxy_Type) {
        PyObject *result = PyDict_New(), *items = NULL;
        if (!result) return NULL;
        if (!PyDict_CheckExact(object)) {
            items = PyMapping_Items(object);
            if (!items) { Py_DECREF(result); return NULL; }
        }
        Py_ssize_t position = 0, index = 0;
        PyObject *key, *value;
        while (1) {
            if (items) {
                if (index >= PyList_GET_SIZE(items)) break;
                PyObject *pair = PyList_GET_ITEM(items, index++);
                if (!PyTuple_CheckExact(pair) || PyTuple_GET_SIZE(pair) != 2) {
                    Py_DECREF(items); Py_DECREF(result); return Py_NewRef(Py_NotImplemented);
                }
                key = PyTuple_GET_ITEM(pair, 0); value = PyTuple_GET_ITEM(pair, 1);
            } else if (!PyDict_Next(object, &position, &key, &value)) break;
            if (!PyUnicode_CheckExact(key)) {
                Py_XDECREF(items); Py_DECREF(result); return Py_NewRef(Py_NotImplemented);
            }
            if (PyUnicode_GET_LENGTH(key) == 0) {
                Py_XDECREF(items); Py_DECREF(result);
                PyErr_SetString(PyExc_ValueError, "decision detail keys must be non-empty strings"); return NULL;
            }
            PyObject *checked_key = freeze_tree(key, depth + 1);
            if (!checked_key || checked_key == Py_NotImplemented) {
                Py_XDECREF(items); Py_DECREF(result); return checked_key;
            }
            Py_DECREF(checked_key);
            PyObject *frozen = freeze_tree(value, depth + 1);
            if (!frozen || frozen == Py_NotImplemented) {
                Py_XDECREF(items); Py_DECREF(result); return frozen;
            }
            int status = PyDict_SetItem(result, key, frozen);
            Py_DECREF(frozen);
            if (status < 0) { Py_XDECREF(items); Py_DECREF(result); return NULL; }
        }
        Py_XDECREF(items);
        PyObject *proxy = PyDictProxy_New(result);
        Py_DECREF(result);
        return proxy;
    }
    return Py_NewRef(Py_NotImplemented);
}

static PyObject *freeze_json(PyObject *self, PyObject *object) { return freeze_tree(object, 0); }

static PyMethodDef methods[] = {
    {"evaluate_counts", evaluate_counts, METH_VARARGS, "Exact intermediate discard evaluation."},
    {"leaf", leaf, METH_VARARGS, "Evaluate a validated next-draw One-Ply Expectation leaf exactly."},
    {"clear_caches", clear_caches, METH_NOARGS, "Clear One-Ply Expectation native structure caches."},
    {"cache_info", cache_info, METH_NOARGS, "Report native cache counters and fixed memory."},
    {"retained_size", retained_size, METH_O, "Conservative private search-object byte accounting."},
    {"freeze_json", freeze_json, METH_O, "Freeze exact built-in JSON trees or return NotImplemented."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_one_ply_native", NULL, -1, methods};
PyMODINIT_FUNC PyInit__one_ply_native(void) { return PyModule_Create(&module); }
