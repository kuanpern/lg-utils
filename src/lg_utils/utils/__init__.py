
def generic_node_factory(agent, key_mapping: dict[str, str] = None):
    """Factory function to create a generic node for a LangGraph StateGraph.

    Args:
        agent: The agent function to be executed within the node.
        key_mapping: An optional dictionary to map output keys from the agent
            to keys in the LangGraph state. If None, the agent's
            output is directly used to update the state.

    Returns:
        A node function suitable for use in a LangGraph StateGraph.
    """

    def node(state):
        output = agent(state)
        if key_mapping is not None:
            updates = {key_mapping[k]: v for k, v in output.items() if k in key_mapping}
        else:
            updates = output
        # end if

        return updates
    # end def
    return node
# end def