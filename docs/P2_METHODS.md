# P2 same-information neural controls

This phase is an engineering comparison, not a claimed reproduction of a
specific paper's scores or a new architecture. It uses the frozen P0 QRF input
contract and seven quantiles (0.025/0.05/0.10/0.50/0.90/0.95/0.975).

## Input and chronology

Every model receives the same canonical QRF fields and the same observed
24-hour history/NWP curve, with the same missingness indicators. After final
train-only imputation, neural inputs equal the saved QRF values exactly before
an invertible train-only standardization. The shape is `(origin,24,128/129)`:
the 24 positions represent forecast leads, and each contains the same columns
as that lead's QRF row. Repeated history/full-weather-curve columns contain no
new information. The network does not receive other origins' labels, longer
history, retrieval targets, new weather products or true future ramps.

Calendar phases remain:

| Dataset | Proper training | Calibration seed for validation | Validation | Evaluation |
|---|---|---|---|---|
| GEFCom2014 | 2012 | Jan–Mar 2013 | Apr–Jun 2013 | Jul–Dec 2013 |
| GEFCom2012 | Jul 2009–Dec 2010 | Jan–Mar 2011 | Apr–Jun 2011 | Jul 2011–27 Jun 2012 |

These are calendar stages; complete-history/NWP eligibility and complete-label
training/scoring masks remain unchanged. Evaluation calibration starts with
arrived validation labels and then updates with completed prior evaluation days.
The latest power/NWP information uses the same P0 zero-latency and inferred
competition-clock assumptions, not newly validated real-world release times.

## Networks and quantile head

The MLP shares weights over leads. TCN uses four residual causal convolutions
with dilations 1/2/4/8; its receptive field spans 31 lead positions. GRU uses
one unidirectional hidden48 layer, with a fresh zero state for each origin.
These temporal modules encode the known forecast-lead sequence; the existing
history vector remains context. There is no separate extended history encoder,
future-power teacher forcing, or joint scenario-generation claim.

All backbones use the same quantile head: an unconstrained base plus cumulative
positive softplus increments, clipped to physical bounds [0,1]. This permits
exact zero/one and ties while guaranteeing nondecreasing raw quantiles. Biases
are initialized using only the relevant training partition's empirical
quantiles. All heads use unweighted mean pinball, with no ramp resampling or
additional width/smoothing/conditional-coverage penalties.

Architectures have similar, not identical, parameter counts. Their normalization,
recurrence/convolution and optimization behavior are bundled differences; an
MLP comparison is informative but not a complete causal isolation of temporal
structure. Each uses the same optimizer and maximum training budget.

## Selection and calibration

Epoch selection uses a chronological 80/20 split *within proper training*.
Its imputer/scaler are fitted only on the optimization prefix. A final model is
then initialized again at the same seed and fitted on all proper-training
labels for the chosen number of epochs. Calibration, external validation and
evaluation labels never stop or train the network. Three seeds are all reported;
none is selected and their predictions are not ensembled.

Raw and calibrated forecasts are both evaluated. Per-lead signed rolling84 CQR
is exactly the P0 rule, including arrived complete-day labels, finite ranks,
physical clipping and midpoint collapse. Independent-level CQR can break
cross-level nesting even when raw neural quantiles are noncrossing; this is
reported without silently applying an extra correction unavailable to QRF.

The joint gate and architecture selection rule were fixed in
`route_a/p2_protocol.json`. Raw WIS/pinball and the MLP controls must be inspected
alongside calibrated coverage/width, not replaced by a single overall score.
All dates are previously viewed development evidence. No exact full-CDF CRPS or
fresh-holdout/deployment/general-model-family claim follows from this phase.

## Runtime references

The implementation uses pinned CPU PyTorch 2.8.0 APIs:
[GRU](https://docs.pytorch.org/docs/2.8/generated/torch.nn.GRU.html),
[Conv1d](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Conv1d.html),
and [deterministic algorithms](https://docs.pytorch.org/docs/2.8/generated/torch.use_deterministic_algorithms.html).
Environment and actual optimizer witnesses are documented in P2_COMPUTE_ENV.md.
