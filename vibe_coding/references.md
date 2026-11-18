# 参考来源与延伸阅读

以下资料用于支撑本文关于模型生成、检索增强、工具调用、Agent、代码评测与风险治理的概念。Vibe Coding 本身仍缺少统一的学术定义，因此本文将其作为工程实践术语，而不是既定理论。

## 基础模型与上下文

1. Vaswani, A. et al. **Attention Is All You Need.** NeurIPS 2017.  
   <https://arxiv.org/abs/1706.03762>
2. Ouyang, L. et al. **Training Language Models to Follow Instructions with Human Feedback.** NeurIPS 2022.  
   <https://arxiv.org/abs/2203.02155>
3. Liu, N. F. et al. **Lost in the Middle: How Language Models Use Long Contexts.** TACL 2024.  
   <https://arxiv.org/abs/2307.03172>

## 检索与工具使用

4. Lewis, P. et al. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.** NeurIPS 2020.  
   <https://arxiv.org/abs/2005.11401>
5. Yao, S. et al. **ReAct: Synergizing Reasoning and Acting in Language Models.** ICLR 2023.  
   <https://arxiv.org/abs/2210.03629>
6. Schick, T. et al. **Toolformer: Language Models Can Teach Themselves to Use Tools.** NeurIPS 2023.  
   <https://arxiv.org/abs/2302.04761>
7. Model Context Protocol. **Specification.**  
   <https://modelcontextprotocol.io/specification/latest>

## 代码生成与评测

8. Chen, M. et al. **Evaluating Large Language Models Trained on Code.** arXiv 2021.  
   <https://arxiv.org/abs/2107.03374>
9. Jimenez, C. E. et al. **SWE-bench: Can Language Models Resolve Real-World GitHub Issues?** ICLR 2024.  
   <https://arxiv.org/abs/2310.06770>
10. Yang, J. et al. **SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering.** arXiv 2024.  
    <https://arxiv.org/abs/2405.15793>

## 风险治理与可信工程

11. NIST. **AI Risk Management Framework.**  
    <https://www.nist.gov/itl/ai-risk-management-framework>
12. OWASP. **Top 10 for Large Language Model Applications.**  
    <https://owasp.org/www-project-top-10-for-large-language-model-applications/>
13. ACM. **Artifact Review and Badging.**  
    <https://www.acm.org/publications/policies/artifact-review-and-badging-current>

## 阅读提示

- 论文给出方法与实验证据，不自动构成对“AI 已具备通用软件工程能力”的证明。
- 协议文档描述接口与语义，不保证具体实现的权限和安全性。
- 工程治理框架提供风险分类和流程，不能替代领域专家、伦理审查或法律意见。
- 引用生成模型提供的来源前，应打开原文核对标题、作者、年份和结论。
