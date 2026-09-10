# Freshness-first edge scheduling: corrected live analysis

The four immutable source cells completed with 300 transmitted frames each. Edge publication is distinct from authoritative map installation.

| action | policy | installed | intentional supersession | expired | true timeout | median AoI | time-weighted AoI |
|---:|---|---:|---:|---:|---:|---:|---:|
| 50 | LATEST_ONLY_NO_EXPIRY | 259 | 36 | 4 | 1 | 278.7 ms | 388.6 ms |
| 50 | LATEST_ONLY_25_MS | 219 | 18 | 63 | 0 | 239.9 ms | 342.7 ms |
| 71 | LATEST_ONLY_NO_EXPIRY | 266 | 33 | 1 | 0 | 217.2 ms | 287.8 ms |
| 71 | LATEST_ONLY_25_MS | 260 | 11 | 29 | 0 | 189.0 ms | 268.9 ms |

The 25 ms expiry ceiling reduced time-weighted AoI by 46.0 ms (11.8%) for action 50 and 18.8 ms (6.6%) for action 71.

It is selected provisionally because freshness is the primary map objective. The cost is fewer installations and more expired, already-transmitted work. No cell installed within 100 ms, so this is not a 100 ms service-readiness result.

Correction: the source aggregate label `intentional_non_install_terminals` included deadline expiry as well as intentional supersession. This analysis separates the unchanged raw counters; no live record was rewritten.
