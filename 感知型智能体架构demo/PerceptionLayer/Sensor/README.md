# 传感器(Sensor)
    传感器监听环境中的一系列事件,所以本质上传感器是一组监听器
    监听器就是在一个事件流/事件总线/消息总线,监听是否存在监听器内部指定要监听的内容,每一个监听器仅可以监听一个主题
    并将其消费将其转化为内部信号
    监听器和事件源完全解耦互相不感知,
    因此我们需要定义一个装饰器或者一个隐式接口一个协议来规范事件源是什么?
    例如使用的时候,我们可以定义
    @event_source(event_name="some_event") 
    def some_event(self):
        return {
            "data": {}
        }
    bus = EventBus(bus=InMemoryEventBus())
    bus.send(some_event) 
