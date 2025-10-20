"这里只是演示了下事件驱动的思路,具体肯定不能这么做,还是需要生成组件的"
from typing import Annotated, TypedDict,List,Optional
import uuid
from langgraph.graph import StateGraph, add_messages
from langchain_core.runnables import Runnable, RunnableConfig

from langgraph.types import Command
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langgraph.types import interrupt
from langchain_openai import ChatOpenAI
llm=ChatOpenAI(
    model_name='Qwen/Qwen3-Coder-30B-A3B-Instruct',
    api_key='sk-lbzsbznhrwackhgeybltsfsghkkllfsafzgttvpvnlgpajol',
    base_url='https://api.siliconflow.cn/v1/',
)
class Event(TypedDict):
    event_name:str
    data:dict
    is_remove: bool=True
# 添加事件的归约函数,做了简单的删除处理
def add_evnet(
    left: List[Event],
    right: List[Event],
) -> List[Event]:
    remove_events = []
    for event in right:
        if event.get('is_remove',False):  
            remove_events.append(event)
    for event in remove_events:
        right.remove(event)
        if event in left:
            left.remove(event)
    
    return left+right

# 删除函数,将事件的is_remove设置为True
def remove_evnet(e:Event):
    e['is_remove']=True

class EventState(TypedDict):
    eventBus: Annotated[List[Event],add_evnet]
    collectedEvent:List[Event]
    
def listener(state: EventState,config:RunnableConfig) -> EventState:
    events = state['eventBus']
    new_events = []
    for event in events:
        if event.get('event_name','') == config['metadata']['event_name']:
            remove_evnet(event)
            new_events.append(event)
    if new_events:
       return Command(
            goto='LLM',
            update={
                "collectedEvent": new_events
            }
        )
    event=interrupt("等待事件的触发")
    return Command(
        goto='listener',
        update={
            "eventBus": event
        }
    )

def LLM(state: EventState) -> EventState:
    events=state['collectedEvent']
    response=llm.stream([SystemMessage(content=f"用户现在在处理信息,根据用户在处理的信息判断是否需要帮助,若用户提及需要帮忙,发出疑问的时候,给予帮助,反之不要做任何的事情可以输出`等待`"),HumanMessage(content=f"{events}")])
    for chunk in response:
        print(chunk.content)
    return state
graph = StateGraph(EventState)
graph.add_node(listener)
graph.add_node(LLM)
graph.add_edge("LLM", "listener")
graph.set_entry_point("listener")
app=graph.compile()
config={"metadata":{"event_name":"写作"}}

def run_interactive_demo():
    """运行交互式演示（带控制台输入）"""
    print("=" * 60)
    print("🤖 感知型智能体交互式演示")
    print("=" * 60)
    print("\n📋 使用说明:")
    print("1. 输入格式: event_name:data")
    print("2. 输入 'quit' 退出程序")
    print("-" * 60)

    # 初始状态：eventBus 里放用户输入的事件
    initial_state = {"eventBus": []}

    state = initial_state
    try:
        while True:
            # ------ 关键：阻塞读用户输入 ------
            user_raw = input("\n>>> ").strip()
            if not user_raw:            # 空输入直接跳过
                continue
            if user_raw.lower() == "quit":
                print("👋 再见！")
                break

            # ------ 解析 event_name:data ------
            if ":" not in user_raw:
                print("⚠️  格式错误，请用 event_name:data")
                continue
            event_name, data = user_raw.split(":", 1)
            event_name, data = event_name.strip(), data.strip()

            # ------ 生成事件并更新状态 ------
            new_event = {"event_name": event_name, "data": {"data":data}}
            state["eventBus"].append(new_event)   # 把事件塞进总线

            # ------ 调用你的智能体 ------
            try:
                state = app.invoke(state, config=config) or state
            except KeyboardInterrupt:
                print("\n\n⚠️  用户中断程序")
                break
            except Exception as e:
                print(f"❌ 智能体执行错误: {e}")
                # 出错就把刚才塞进去的事件弹回，避免无限累积
                state["eventBus"].pop()
                
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断程序")
    except Exception as e:
        print(f"❌ 顶层错误: {e}")

if __name__ == "__main__":
    run_interactive_demo()



