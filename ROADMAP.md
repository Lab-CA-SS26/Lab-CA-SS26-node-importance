# Graphs.jl Contribution Roadmap: KADABRA Centrality

This roadmap details the exact steps and technical guidelines required to contribute your KADABRA betweenness centrality implementation to the official [Graphs.jl](https://github.com/JuliaGraphs/Graphs.jl) open-source library.

---

## 🛠️ Step 1: Package Structure Integration

To merge the standalone files into the local clone of the `Graphs.jl` codebase:

1. **Move Code File**:
   Copy the core `kadabra.jl` implementation into the centrality folder of `Graphs.jl`:
   ```bash
   cp kadabra.jl Graphs.jl/src/centrality/kadabra.jl
   ```

2. **Register imports and exports**:
   - Open `Graphs.jl/src/centrality/centrality.jl` (or `Graphs.jl/src/Graphs.jl`) and add:
     ```julia
     include("centrality/kadabra.jl")
     ```
   - Export the function in the same module file:
     ```julia
     export kadabra_centrality
     ```

3. **Integrate Unit Tests**:
   - Copy the test cases from `test_kadabra.jl` and append them to the existing centrality tests in `Graphs.jl/test/centrality.jl`.
   - Ensure the tests execute within the Graphs.jl test framework without conflicts.

---

## 🎨 Step 2: Code Style & Linting Compliance

`Graphs.jl` maintains a strict, unified code style and type safety framework:

1. **Auto-Formatting (`JuliaFormatter.jl`)**:
   Run the Julia formatter to enforce the package-wide coding guidelines:
   ```julia
   using JuliaFormatter
   format_file("src/centrality/kadabra.jl")
   ```

2. **Type-Stability & Static Analysis (`JET.jl`)**:
   Ensure that the sampler, backtracking helpers, and diameter estimation are completely type-stable and do not have type-instability allocations:
   ```julia
   using JET
   report_package("Graphs")
   ```

---

## 🐙 Step 3: Git & GitHub Workflow

Follow the fork-and-pull model to propose your contribution:

1. **Fork the Repository**:
   Fork the main [JuliaGraphs/Graphs.jl](https://github.com/JuliaGraphs/Graphs.jl) repository on GitHub to your personal account.

2. **Clone & Branch**:
   Clone your personal fork locally and create a dedicated branch:
   ```bash
   git clone https://github.com/your-username/Graphs.jl.git
   cd Graphs.jl
   git checkout -b feature/kadabra-centrality
   ```

3. **Verify Local Test Execution**:
   Run the full package test suite to verify there are no regressions:
   ```bash
   julia --project -e 'using Pkg; Pkg.test()'
   ```

4. **Commit & Push**:
   Commit the integrated changes and push to your fork:
   ```bash
   git add src/ centrality/ test/
   git commit -m "Implement KADABRA betweenness centrality algorithm"
   git push origin feature/kadabra-centrality
   ```

---

## 🚀 Step 4: Open a Pull Request (PR)

1. Go to the original [JuliaGraphs/Graphs.jl](https://github.com/JuliaGraphs/Graphs.jl) repository.
2. Click **New Pull Request** and choose your branch `feature/kadabra-centrality` as the source.
3. Write a professional PR description, detailing:
   - **Performance Results**: Showcase the benchmark comparison (Julia `0.247`s vs C++ `0.199`s) to prove the implementation's efficiency.
   - **Enhancements**: Highlight `AllCCUpperBound` diameter estimation and modern loop-index-based threading safety.
   - **References**: Cite the original KADABRA paper (Borassi & Natale 2019) and the diameter bounding paper (Borassi et al. 2015).
