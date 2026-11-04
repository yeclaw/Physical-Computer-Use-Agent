# Recorded demonstrations and accounting

These runs describe continuous use and learning on one reference setup. Knowledge accumulates within each condition. They are not a frozen benchmark across unseen devices.

## Continuous chat

- Initial support: the complete 39-key demonstration catalog and previously accumulated skills.
- Agent: `gpt-6-astra`, `xhigh`, default service tier, through Codex app-server.
- Interaction: cameras observe the display; a robot enters text using a real keyboard. There were 24 sent messages totaling 634 characters.
- One draft was interrupted when the operator switched windows. The agent inspected the mismatch, used Backspace, completed the draft, checked it, and sent it. This is a recorded repair with an operator-reported external cause.
- The complete run lasted approximately 3 hours 14 minutes. The presentation video is edited and time-compressed; it is not a real-time playback.

The model made 160 completed decisions. Runtime executed 53 programs: 25 draft programs, 24 Return programs, one Backspace program, two low-position exits, and one read-only listing. Logical model decisions and physical executions have different counts.

| Native usage | Total | Mean per sent message |
|---|---:|---:|
| Total input plus output, including cached input | 27,559,037 | 1,148,293.21 |
| Cached input | 24,982,016 | 1,040,917.33 |
| Uncached input | 2,531,885 | 105,495.21 |
| Output, including reasoning-output accounting | 45,136 | 1,880.67 |

Cached input is a subset of input; reasoning output is a subset of output. Input cache hit rate is 90.80%. These are processing counters, not a monetary bill. The mean includes startup, observation, planning, repair, knowledge operations, and waiting. It excludes earlier training and the other chat service's token usage.

The [160 decision usage rows](../examples/results/chat-decision-usage.csv) sum to the [aggregate summary](../examples/results/chat-summary.json). Raw screen/account/runtime records are kept outside the source release.

## Partial demonstration learning

The condition supplied six letter demonstrations (`r`, `u`, `f`, `j`, `v`, `n`), Backspace/Space demonstrations, and a Return program. It used a separate knowledge store from the complete-demonstration chat run. General keyboard/task priors and declared device information were available.

The 21-request curriculum asked for `g`, `h`, `b`, `e`, `l`, `o`, and `hello`, each three times with revisits. All 21 requested strings were ultimately accepted, including retries and corrections. This is not a 21/21 first-attempt result or evidence that the whole keyboard was learned.

Recorded examples include:

- `g`: inspect supplied `f`/`j` trajectories, propose a pose, observe a mistaken `h`, adjust position/depth, save a candidate, and call it again.
- `e`: inspect supplied `r`/`u` references, propose a position from their spacing, and calibrate depth through feedback.
- `o`: use a `u` reference and prior spacing estimate, later revise the local press/release path, save it, and actually reuse that same program in `hello`.
- Repeated `l`: a Lesson records that similar sampled endpoints do not ensure a single character. Backspace repairs the current text while the skill's repetition issue remains open.

The chat video uses the complete-demonstration condition. It is not presented as the direct output of the six-letter demonstration run.

## Remaining evaluations

Fresh zero-demonstration word learning, moved-keyboard adaptation, and independent device replications remain to be completed. The recorded outcomes do not establish a controlled speed/token improvement over another agent method.
