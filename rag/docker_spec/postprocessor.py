from collections import deque
from collections.abc import Callable
from copy import deepcopy
from typing import Literal, Optional, Sequence

from llama_index.core.bridge.pydantic import Field, field_validator
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import (
    BaseNode,
    MetadataMode,
    NodeRelationship,
    NodeWithScore,
    QueryBundle,
    TextNode,
)


def dedup_nodes_with_score(nodes: Sequence[NodeWithScore]) -> list[NodeWithScore]:
    hash_dict: dict[str, NodeWithScore] = dict()
    for node in nodes:
        hash = node.node.hash
        if (hash in hash_dict) and (hash_dict[hash].get_score() > node.get_score()):
            continue
        hash_dict[hash] = node
    return sorted(list(hash_dict.values()), key=lambda x: x.get_score(), reverse=True)


class MergeConsecutiveNodesPostprocessor(BaseNodePostprocessor):
    @classmethod
    def class_name(cls) -> str:
        return "MergeConsecutiveNodesPostprocessor"

    @staticmethod
    def _join_nodes(
        prev_node_with_score: NodeWithScore, next_node_with_score: NodeWithScore
    ) -> NodeWithScore:
        prev_node_with_score = deepcopy(prev_node_with_score)
        prev_node = prev_node_with_score.node
        assert isinstance(prev_node, TextNode)
        prev_node_start_end = prev_node.get_node_info()
        next_node = next_node_with_score.node
        assert isinstance(next_node, TextNode)
        next_node_start_end = next_node.get_node_info()
        if (prev_node_start_end.get("end") is not None) and (
            next_node_start_end.get("start") is not None
        ):
            new_content = (
                prev_node.get_content(metadata_mode=MetadataMode.NONE)
                + next_node.get_content(metadata_mode=MetadataMode.NONE)[
                    prev_node_start_end["end"] - next_node_start_end["start"] :
                ]
            )
        else:
            new_content = (
                prev_node.get_content(metadata_mode=MetadataMode.NONE)
                + "\n"
                + next_node.get_content(metadata_mode=MetadataMode.NONE)
            )
        new_score = max(
            next_node_with_score.get_score(),
            prev_node_with_score.get_score(),
        )
        prev_node.set_content(new_content)
        prev_node_with_score.score = new_score
        if (NodeRelationship.NEXT in next_node.relationships) and (
            next_node.next_node is not None
        ):
            prev_node.relationships[NodeRelationship.NEXT] = next_node.next_node
        else:
            prev_node.relationships.pop(NodeRelationship.NEXT)
        prev_node.end_char_idx = next_node_start_end.get("end")
        return prev_node_with_score

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> list[NodeWithScore]:
        """Postprocess nodes."""
        nodes = dedup_nodes_with_score(nodes=nodes)
        all_nodes: dict[str, NodeWithScore] = {
            node.node.node_id: node for node in nodes
        }
        node_id_set = set(all_nodes.keys())
        new_list = []
        while node_id_set:
            node_id = node_id_set.pop()
            node_with_score = all_nodes[node_id]
            node = node_with_score.node
            node_deque = deque([node_id])
            while True:
                if NodeRelationship.PREVIOUS in node.relationships:
                    prev_node_info = node.prev_node
                    if prev_node_info is not None:
                        prev_node_id = prev_node_info.node_id
                        if prev_node_id in all_nodes:
                            prev_node_with_score = all_nodes[prev_node_id]
                            assert (
                                (
                                    NodeRelationship.NEXT
                                    in prev_node_with_score.node.relationships
                                )
                                and (prev_node_with_score.node.next_node is not None)
                                and (
                                    prev_node_with_score.node.next_node.node_id
                                    == node_deque[0]
                                )
                            )
                            node_deque.appendleft(prev_node_id)
                            node_with_score = self._join_nodes(
                                prev_node_with_score=prev_node_with_score,
                                next_node_with_score=node_with_score,
                            )
                            node = node_with_score.node
                            node_id_set.discard(prev_node_id)
                            continue
                break
            while True:
                if NodeRelationship.NEXT in node.relationships:
                    next_node_info = node.next_node
                    if next_node_info is not None:
                        next_node_id = next_node_info.node_id
                        if next_node_id in all_nodes:
                            next_node_with_score = all_nodes[next_node_id]
                            assert (
                                (
                                    NodeRelationship.PREVIOUS
                                    in next_node_with_score.node.relationships
                                )
                                and (next_node_with_score.node.prev_node is not None)
                                and (
                                    next_node_with_score.node.prev_node.node_id
                                    == node_deque[-1]
                                )
                            )
                            node_deque.append(next_node_id)
                            node_with_score = self._join_nodes(
                                prev_node_with_score=node_with_score,
                                next_node_with_score=next_node_with_score,
                            )
                            node = node_with_score.node
                            node_id_set.discard(next_node_id)
                            continue
                break
            new_list.append(node_with_score)
        return new_list


class PrevNextNodePostprocessor(BaseNodePostprocessor):
    """
    Previous/Next Node post-processor.

    Allows users to fetch additional nodes from the document store,
    based on the relationships of the nodes.

    NOTE: this is a beta feature.

    Args:
        docstore (BaseDocumentStore): The document store.
        num_nodes (int): The number of nodes to return (default: 1)
        mode (str): The mode of the post-processor.
            Can be "previous", "next", or "both.

    """

    id_node_fn: Callable[[str], BaseNode | None]
    num_nodes: int = Field(default=1)
    mode: str = Field(default="next")

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, v: str) -> str:
        """Validate mode."""
        if v not in ["next", "previous", "both"]:
            raise ValueError(f"Invalid mode: {v}")
        return v

    @classmethod
    def class_name(cls) -> str:
        return "PrevNextNodePostprocessor"

    @staticmethod
    def get_for_back_nodes(
        node_with_score: NodeWithScore,
        num_nodes: int,
        id_node_fn: Callable[[str], BaseNode | None],
        direction: Literal["prev", "next"],
    ) -> dict[str, NodeWithScore]:
        """Get forward nodes."""
        node = node_with_score.node
        nodes: dict[str, NodeWithScore] = {node.node_id: node_with_score}
        cur_count = 0
        # get forward and backward nodes in an iterative manner
        while cur_count < num_nodes:
            if direction == "next":
                if NodeRelationship.NEXT not in node.relationships:
                    break
                new_node_info = node.next_node
            elif direction == "prev":
                if NodeRelationship.PREVIOUS not in node.relationships:
                    break
                new_node_info = node.prev_node

            if new_node_info is None:
                break
            new_node_id = new_node_info.node_id
            new_node = id_node_fn(new_node_id)
            if new_node is None:
                break

            next_node_info = node.next_node
            if next_node_info is None:
                break

            nodes[new_node.node_id] = NodeWithScore(node=new_node)
            node = new_node
            cur_count += 1
        return nodes

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> list[NodeWithScore]:
        """Postprocess nodes."""
        all_nodes: dict[str, NodeWithScore] = {}
        for node in nodes:
            all_nodes[node.node.node_id] = node
            if self.mode == "next":
                all_nodes.update(
                    self.get_for_back_nodes(
                        node,
                        self.num_nodes,
                        id_node_fn=self.id_node_fn,
                        direction="next",
                    )
                )
            elif self.mode == "previous":
                all_nodes.update(
                    self.get_for_back_nodes(
                        node,
                        self.num_nodes,
                        id_node_fn=self.id_node_fn,
                        direction="prev",
                    )
                )
            elif self.mode == "both":
                all_nodes.update(
                    self.get_for_back_nodes(
                        node,
                        self.num_nodes,
                        id_node_fn=self.id_node_fn,
                        direction="next",
                    )
                )
                all_nodes.update(
                    self.get_for_back_nodes(
                        node,
                        self.num_nodes,
                        id_node_fn=self.id_node_fn,
                        direction="prev",
                    )
                )
            else:
                raise ValueError(f"Invalid mode: {self.mode}")

        all_nodes_values: list[NodeWithScore] = list(all_nodes.values())
        sorted_nodes: list[NodeWithScore] = []
        for node in all_nodes_values:
            # variable to check if cand node is inserted
            node_inserted = False
            for i, cand in enumerate(sorted_nodes):
                node_id = node.node.node_id
                # prepend to current candidate
                prev_node_info = cand.node.prev_node
                next_node_info = cand.node.next_node
                if prev_node_info is not None and node_id == prev_node_info.node_id:
                    node_inserted = True
                    sorted_nodes.insert(i, node)
                    break
                # append to current candidate
                elif next_node_info is not None and node_id == next_node_info.node_id:
                    node_inserted = True
                    sorted_nodes.insert(i + 1, node)
                    break

            if not node_inserted:
                sorted_nodes.append(node)

        return sorted_nodes
