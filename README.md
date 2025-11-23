# two-and-seven-joints
This demo generates synthetic robot data (two velocities q_
d(1,2), their accelerations q¨ (1,2)
,
a command torque u, and a measured torque τ_meas). From rolling windows of length W,
three quantities are estimated: (i) coherence, (ii) reactivity, and (iii) directed coupling. These
are combined into a complex strategy operator Ξ = ℜΞ + iℑΞ, which is projected along four
decision axes (execute / guard / stop / pivot) using a Softmax policy. A secondary script
streams a multi-joint feed, performs real-time window updates, and optionally exports a video.
