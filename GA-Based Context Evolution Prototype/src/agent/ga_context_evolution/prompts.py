from langchain_core.prompts import ChatPromptTemplate

# 1. Target Profiling (目标画像)
TARGET_PROFILING_PROMPT = ChatPromptTemplate.from_template(
    """你是一位专业的上下文工程专家（Context Engineer）。你的目标是分析用户的目标描述，并创建一个详细的“理想上下文画像（Ideal Context Profile）”。
    
    用户的目标描述：
    {target_description}
    
    请输出一份为了完成该任务所需的理想上下文的特征信息,这些信息并不作为生成该任务的上下文,而是反向生成该任务目标完成所需要的可能元特征或者说该任务潜在的目标是什么?
    
    请直接输出最大数量为{max_profile}的特征 
    参考: 
    目标: 1+1=?
    特征: 输出结果约束,输出风格约束,输出长度约束
    """
)

# 2. Gene Extraction (基因提取/分解)
GENE_EXTRACTION_PROMPT = ChatPromptTemplate.from_template(
    """
    你的任务是针对:“理想上下文画像"中的特征描述:
    {ideal_context_profile}
    针对当前任务{target_description},
    对每个独立的特征,生成 {num_chunks} 个不同的“提示词指令(Prompt Instructions)”或“约束条件”。
    
    【关键要求】
    1. 生成的内容必须是**控制生成的指令**，而不是内容本身。
       - 错误示例（内容型）："Rust通过所有权确保安全..."
       - 正确示例（指令型）："请用通俗的语言解释概念"、"回答必须包含具体的代码示例"、"将回复限制在3句话以内"。
    2. 每个指令都应该是独立的，并且能够引导模型生成符合该特征的内容。
    3. 每个特征生成的指令需要具有显著差异（例如：语气、格式、侧重点的不同）。

    【重要】输出的JSON字典的Key必须严格与上述“理想上下文画像”中的特征描述保持一致，不得修改、简化或翻译。
    """
)

# 3. Context Generation (上下文生成)
CONTEXT_GENERATION_PROMPT = ChatPromptTemplate.from_template(
    """
    你是一个智能助手。请严格遵循以下**系统指令**来完成用户的目标。

    【系统指令 / 角色设定】
    {context_content}

    【用户目标】
    {target_description}

    请直接输出结果，不要包含任何元数据或解释。
    """
)

# 4. Reverse Profiling (反向画像/特征提取)
REVERSE_PROFILING_PROMPT = ChatPromptTemplate.from_template(
    """
    你是一位严格的文本分析师。请分析以下“生成内容”，并提取其显著特征。

    【用户目标】
    {target_description}

    【生成内容】
    {generated_result}

    请提取出该生成内容实际表现出的 3-5 个关键特征。
    请保持特征描述简练且具体（例如：“包含代码示例”、“语气幽默”、“回答简短”、“直接输出结果”）。
    """
)

# 5. Evolution Analysis (进化分析)
EVOLUTION_ANALYSIS_PROMPT = ChatPromptTemplate.from_template(
    """
    你是一个进化分析师。请分析当前的进化结果，并给出改进建议。
    
    当前代数：{generation_round}
    最高分：{max_score}
    平均分：{avg_score}
    
    请简要分析当前种群的多样性和收敛情况。
    """
)

# 6. Gene Mutation (基因突变)
GENE_MUTATION_PROMPT = ChatPromptTemplate.from_template(
    """
    你是一个基因编辑器。你的任务是对给定的“Prompt片段”进行微小的突变（修改）。
    
    【原始片段】
    {original_content}
    
    【突变要求】
    1. 保持原始片段的核心语义和功能不变。
    2. 对措辞、语气或细节进行微调。例如：换一种更强烈的说法、更委婉的说法、或者精简/扩充描述。
    3. 使得这个片段在表达上与原文有所区分，但目标一致。
    
    请直接输出突变后的内容，不要包含任何解释或引号。
    """
)
