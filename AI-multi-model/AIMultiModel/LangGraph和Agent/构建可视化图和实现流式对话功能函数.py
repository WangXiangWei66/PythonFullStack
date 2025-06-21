# 将状态图转化为Mermaid流程图并保存PNG图像
def draw_graph(graph, png_file):  # 获取底层的图结构图像
    try:
        image = graph.get_graph().draw_mermaid_png()  # 生成Mermaid格式的流程图并转换为PNG
        with open(png_file, "wb") as f:
            f.write(image)
    except Exception as e:  # 如果图生成失败，捕获并打印可能的错误
        print(e)


# 处理用户输入并以流式方式显示AI回复
# def loop_graph_invoke(graph, user_input: str):
#     '''循环调用这个流程图，让AI可以一直和用户对话'''
# for chunk in graph.stream({'message': [('user', user_input)]}):  #以流模式执行图，逐步生成回复
#     for value in chunk.values():
#         print('AI机器人:', value['message'][-1].content)

# stream_mode = "values"
# stream_mode = “values”:这意味着该方法将会直接返回事件中的值，而不是整个事件对象，这使得处理过程更加简洁，特别是当你只关心事件的具体内容而非元数据时
# events = graph.stream({"messages": [("user", user_input)]}, stream_mode="values")  #它还将用户输入封装为了消息格式
# for event in events:
#     event["messages"][-1].pretty_print()  #格式化打印最新的AI回复


# 支持传入参数配置
def loop_graph_invoke(graph, user_input: str, config):
    # config：控制图执行的配置（如工具选择策略，超时设置）
    events = graph.stream({'messages': [('user', user_input)]}, config, stream_mode="values")
    for event in events:
        event["messages"][-1].pretty_print()


# 支持空输入的对话处理（如继续之前的思考或工具调用）
def loop_graph_invoke_tools(graph, user_input: str, config):
    if user_input:  # 如果有用户输入，则正常处理
        events = graph.stream(
            {'messages': [('user', user_input)]}, config, stream_mode="values"
        )
        for event in events:
            if "messages" in event:
                event["messages"][-1].pretty_print()
    else:  # 否则传了None作为输入，则会触发工具的连锁调用
        events = graph.stream(None, config, stream_mode="values")
        for event in events:
            if "messages" in event:
                event["messages"][-1].pretty_print()
