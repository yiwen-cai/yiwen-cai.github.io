#import "@preview/basic-resume:0.2.3": *
// Put your personal information here, replacing mine
#let name = "Cai Yiwen"
#let location = "Beijing, China"
#let email = "caiyiwen.cs@foxmail.com"
#let github = "github.com/yiwei-cai"
#let phone = "(+86) 189-2876-9793"
#let personal-site = "yiwen-cai.github.io"


#show: resume.with(
  author: name,
  // All the lines below are optional.
  // For example, if you want to to hide your phone number:
  // feel free to comment those lines out and they will not show.
  location: location,
  email: email,
  github: github,
  phone: phone,
  personal-site: personal-site,
  accent-color: "#26428b",
  // CI runner 上没有 Times New Roman，写了也会静默回退到 Libertinus Serif，索性写实际生效的
  font: "Libertinus Serif",
  paper: "us-letter",
  author-position: center,
  personal-info-position: center,
)

/*
 * Lines that start with == are formatted into section headings
 * You can use the specific formatting functions if needed
 * The following formatting functions are listed below
 * #edu(dates: "", degree: "", gpa: "", institution: "", location: "")
 * #work(company: "", dates: "", location: "", title: "")
 * #project(dates: "", name: "", role: "", url: "")
 * certificates(name: "", issuer: "", url: "", date: "")
 * #extracurriculars(activity: "", dates: "")
 * There are also the following generic functions that don't apply any formatting
 * #generic-two-by-two(top-left: "", top-right: "", bottom-left: "", bottom-right: "")
 * #generic-one-by-two(left: "", right: "")
 */
== Education

#edu(
  institution: "Beijing University of Posts and Telecommunications (BUPT)",
  location: "Beijing, China",
  dates: dates-helper(start-date: "Sep 2026", end-date: "Present"),
  degree: "Master's degree in Computer Science and Technology (Recommended Admission)",
)

#edu(
  institution: "Beijing University of Posts and Telecommunications (BUPT)",
  location: "Beijing, China",
  dates: dates-helper(start-date: "Sep 2022", end-date: "Jun 2026"),
  degree: "Bachelor's degree in Computer Science and Technology",
)
- GPA: 3.76 / 4.0
- Grade: 89.37 / 100.0
- Rank: 59 / 389 (15.16%)
- English: CET6 552 / 710

== Work Experience

#work(
  title: "LLM High-Performance Inference Framework Scheduling & Operator Optimization Research",
  location: "Beijing, China",
  company: "Tencent Hunyuan AI Infra",
  dates: dates-helper(start-date: "Jul 2026", end-date: "Present"),
)
- Optimized inference framework scheduling and high-performance operators for LLM serving
- Developed operators including Blackwell-architecture BF16 paged KV attention prefill, nvfp4_blockwise_gemm, per_group MXFP8 quantization, and fused_silu_per_token_quant fusion kernel
- Main open-source repository: #link("https://github.com/Tencent/hpc-ops")[Tencent/hpc-ops]
- Part of the contributed code has been merged into the vLLM main branch

== Projects

#work(
  title: "The 10th Huawei ICT Competition China Finals Challenge Track",
  location: "Huazhong University of Science and Technology, Wuhan",
  company: "Huawei",
  dates: dates-helper(start-date: "Apr 2026", end-date: "Apr 2026"),
)
- Ranked 8th among 64 university teams to advance to finals
- Ranked 3rd overall among 16 finalist teams, awarded First Prize
- Achieved best optimization among all teams in the operator migration track as the sole developer
- Result: National First Prize, Best Operator Migration Optimization Award
- Projects (selected):
  - #project(name: link("https://github.com/yiwei-cai/fused_add_rmsnorm_kernel")[fused_add_rmsnorm_kernel])
    - A Triton-based fused RMSNorm and add kernel
    - Multi-row program mapping
    - hidden_size bucketed scheduling
    - 19.28x speedup vs. PyTorch baseline
  - #project(name: link("https://github.com/yiwei-cai/xllm-ictfinal")[xllm-ictfinal])
    - Adapted Qwen3.5 model inference on the xllm inference engine
    - Implemented GDN linear attention adaptation
    - Implemented MTP speculative decoding
    - Resolved KVCache memory anomaly
    - Successfully ran 32k-token input inference


#project(
  name: link("https://github.com/yiwei-cai/cuda-gemm")[cuda-gemm],
  dates: "Jun 2025",
)
- Hand-written 8 CUDA GEMM kernels from scratch, covering FP32 CUDA Core and FP16 Tensor Core WMMA
- Progressive optimizations: Shared Memory Tiling, Bank Conflict Avoidance, Double Buffering, cp.async
- SGEMM v3 achieves 95.1% of cuBLAS (4096²: 37,898 GFLOPS; 8192²: 36,799 GFLOPS)
- HGEMM v3 Tensor Core reaches 213 TFLOPS at 4096², with Nsight Compute/Systems profiling

#work(
  title: "NUS Summer Workshop",
  location: "Singapore",
  company: "National University of Singapore",
  dates: dates-helper(start-date: "Jul 2024", end-date: "Aug 2024"),
)
- Participated in Cloud Computing and Big Data course and completed a group-designed project
- Grade: A+
- Rank: 1st
- Project: #project(name: link("https://github.com/uplion")[UPLION], dates: dates-helper(start-date: "Jul 2024", end-date: "Aug 2024"))
  - A unified platform for integrating various AI agents for comprehensive AI operations and deployment
  - Cloud-native design for scalability and resilience
  - Intelligent task distribution and resource optimization
  - Kubernetes integration for efficient worker node management

#project(name: link("https://github.com/yiwei-cai/fastGNN")[FastGNN], dates: "May 2026")
- A TensorCore-based graph neural network training system
- H100 full-batch GNN
- Custom sparse operator backend
- Voltrix block-sparse SpMM
- Fused3S sparse attention
- Rabbit node reordering
- End-to-end benchmark pipeline
- Validated against DGL/PyG, 1.8x–3.3x speedup

== Awards
#project(
  name: "The 10th Huawei ICT Competition China Finals Challenge Track",
  role: text[First Prize, Best Operator Migration Optimization Award],
  dates: "Apr 2026",
)

#project(
  name: "NUS Summer Workshop",
  role: text[A+],
  dates: "Jul 2024",
)

#project(
  name: "15th Blue Bridge Cup National Software and Information Technology Professional Talent Competition Beijing Region",
  role: text[2#super[nd] Prize],
  dates: "Apr 2024",
)

#project(name: "School-level Scholarship (Junior Year)", dates: "2025")

#project(name: "School-level Scholarship (Sophomore Year)", dates: "2024")

#project(name: "School-level Scholarship (Freshman Year)", dates: "2023")

== Skills
- *Programming Languages*: Python, C/C++, CUDA, Triton, Rust, Go
- *Research Directions*: LLM training/inference optimization, operator optimization, distributed & cloud computing, high-performance computing
