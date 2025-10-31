
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, AnyMessage,SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph ,END
from typing import List,Annotated,cast,Set
from src.prompt import tool_kit_prompt
from src.config import get_think_model,logging,get_model_think_v4
from langgraph.prebuilt import ToolNode
from langgraph.types import Command


# 我现在设计了一个关于模型思考增强的架构具体逻辑是这样子
# 核心
# 1,思考分解,将单纯的思考分解成为若干不同类型的思考方式,将原本的单次长链条思考改为多次的短内容的思考链,
# 2,思考外置,使用工具或者意图识别等方式,总之我们将思考外置出去,并令模型自主选择思考的方式
# 3,思考过程记录,将一次对话中的每一次思考记录下来,并建立思考之间的因果关系,由模型自己选择。最终构建一个思考网络
# 4,每次模型思考的时候根据最相似观点随机的探索并设定探索数量限制。
# 5,定量实行减支策略,减枝策略的方式是判断每个独立的思考在整个任务和网络中的价值。
# 重复上述行为
# 你觉得如何?你有什么建议嘛?

# 补充,LLM的Agent方向,不涉及算法,思考就是让模型输入一段"思考文本"

# 提取 meta_thinks 的 keys 用于 Literal 类型
class ALNode:
    """
    邻接表节点
    每个节点维护：
      - 唯一标识 id（任意可哈希类型）
      - 任意附加数据 payload（字典，可选）
      - 出边集合 successors：set[ALNode]
      - 入边集合 predecessors：set[ALNode]
    支持有向/无向、带权/不带权（边权可放在 payload 或单独 Edge 对象）
    """
    __slots__ = ("id","index" , "payload", "successors", "predecessors")

    def __init__(self, id,index, payload=None):
        self.id = id
        self.index=index
        self.payload = payload or {}
        self.successors: set[ALNode] = set()
        self.predecessors: set[ALNode] = set()

    # -------------------- 基本边操作 --------------------
    def add_successor(self, other: "ALNode", bidirectional=False):
        self.successors.add(other)
        other.predecessors.add(self)
        if bidirectional:
            other.successors.add(self)
            self.predecessors.add(other)

    def remove_successor(self, other: "ALNode", bidirectional=False):
        self.successors.discard(other)
        other.predecessors.discard(self)
        if bidirectional:
            other.successors.discard(self)
            self.predecessors.discard(other)

    # -------------------- 遍历辅助 --------------------
    def dfs(self, visited=None):
        """以当前节点为起点做深度优先，返回生成器"""
        if visited is None:
            visited = set()
        if self in visited:
            return
        visited.add(self)
        yield self
        for nxt in self.successors:
            yield from nxt.dfs(visited)

    # -------------------- 可视化/调试 --------------------
    def to_dict(self):
        return {
            "id": self.id,
            "payload": self.payload,
            "successors": [n.id for n in self.successors],
            "predecessors": [n.id for n in self.predecessors],
        }

    def __repr__(self):
        return f"ALNode({self.id})"


# 
class State(BaseModel):
    """状态类,用于存储状态"""
    messages: Annotated[List[AnyMessage], add_messages] = []
    memory_for_think: Annotated[Set[ALNode], Field(default=[])] = []
    
    think_prompt: Annotated[str, Field(default="")] = ""

async def master(state:State)->State:
    logging.info(f"主节点准备")
    """
    主模型,用于对话的模型和进行决策的模型块
    """
    sys_message=f"""
        <思考>
            关于思考,
            你总是利用工具进行思考而非直接文本推理
            你总是习惯先去想法而非下定论
            每个思考对象总是明确的独立的,它总是最简的指向所思的那个对象的。
            思考的内容总是至少作为一条完整句子可理解的
            一个复杂的思考总是又若干简单的思考迭代而成的。
        </思考>
    """
    history=state.messages
    think_model=get_think_model(tool_kit_prompt)

    response =cast(AIMessage, await think_model.ainvoke([SystemMessage(content=sys_message),*history]))
    logging.info(f"主节点输出结果:content{response.content},tool_call{response.tool_calls}")
    return {"messages": [response]}

async def think(state:State)->State:
    logging.info(f"思考节点预备")
    """思考节点,用于思考"""
    think_prompt=state.think_prompt
    history=state.messages
    think_model=get_model_think_v4()
    base_think_prompt="""
        Agent不会的内容写长,因为如果写得长了,Agent会觉得没有价值;
        Agent用简洁清晰的方式表达当前的思考;
        每一段思考总是简单的;
        当前的输出是Agent的独白,只给Agent自己看，其他人看不到,
    """
    response =cast(AIMessage,await think_model.ainvoke([SystemMessage(content=base_think_prompt),SystemMessage(content=think_prompt)]))
    logging.info(f"思考节点输出结果:content{response.content}")
    return {"messages": [SystemMessage(f"Agent 想到:`{response.content}`")]}

async def tool_call(state:State):
    """工具节点,用于执行工具"""
    if (state.messages[-1].tool_calls):
        logging.info(f"继续思考")
        tool_node = ToolNode(tools=tool_kit_prompt)
        result = await tool_node.ainvoke(state)
        if (result):
            # 提取工具调用结果中的内容作为思考提示
            think_prompt_content = result["messages"][-1].content
            logging.info(f"思考方式{think_prompt_content}")
            return Command(
                    goto="think",
                    update={"think_prompt": think_prompt_content}
                )
    logging.info(f"结束思考")
    return Command(
                goto=END,
            )



graph_build = StateGraph(State)
graph_build.add_node("master",master)
graph_build.add_node("tool_call",tool_call)
graph_build.add_node("think",think) 
graph_build.add_edge(START,"master")
graph_build.add_edge("master","tool_call")
graph_build.add_edge("think","master")



graphv3_5 = graph_build.compile(name="agent").with_config(
            config={
                "recursion_limit": 100
            }
        )