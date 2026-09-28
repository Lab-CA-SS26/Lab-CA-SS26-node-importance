# Final presentation

`presentation.pdf` holds the slides of the final talk on 16 September 2026 (without speaker notes), and
`presentation_demo/` the interactive demo shown in it. Open
`presentation_demo/betweenness_demo.html` in a browser, or use the slides' demo buttons with the folder
kept next to the PDF. An extended version of the demo is hosted at
<https://projectsschmachtin.github.io/betweenness-demo/>.

## The numbers in the slides are not the final ones

After the talk, prompted by the feedback on it, I investigated why our Julia port drew 3–6% more
samples than the authors' C++ reference. The cause was in the port's parallel stopping check:
threads finished their current batch after another thread had already stopped the run, and the
check ignored samples from unfinished batches. Fixing this changed the code, so every KADABRA
measurement was re-run afterwards.

The slides therefore show the numbers from before the fix. For example, the talk reports the Julia
port at 1.04× the reference's samples and 0.98× its time; after the fix it is 1.00× and 0.90×.
The report ([`../report.pdf`](../report.pdf)) contains the current numbers.
