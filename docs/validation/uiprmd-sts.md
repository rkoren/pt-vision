# Sit-to-stand segmentation on UI-PRMD

Run on 2026-09-22 with `ptv datasets eval-sts`. UI-PRMD (public domain, PDDL) has 10 subjects
performing 10 sit-to-stand episodes each, correctly and then deliberately incorrectly, recorded with a
39-marker Vicon set at 100 Hz. Each segmented episode is exactly one repetition. Markers were
projected onto a virtual side camera and run through the same sit-to-stand segmenter used on video.

| set | episodes | exactly one rise detected | zero | multiple |
| --- | --- | --- | --- | --- |
| correct | 100 | 100 | 0 | 0 |
| incorrect | 100 | 100 | 0 | 0 |

Rep counting is solid on clean marker data. There is no timing ground truth in this dataset, so it
says nothing about seat-off or stand accuracy; that comes from COMFI's force plates and from your own
hand-timed recordings.

Reproduce: `uv run ptv datasets pull uiprmd && uv run ptv datasets eval-sts`
(the original UI-PRMD site is gone; files come from the Wayback Machine capture of 2024-07-07).
