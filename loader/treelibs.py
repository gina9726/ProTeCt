### treelibs.py
# Classes for building/analyzing tree hierarchy and hierarchical parameterization.
# Ref: https://arxiv.org/pdf/2007.09898.pdf
# Author: Tz-Ying Wu
###

import numpy as np
import os
from termcolor import colored


class TreeNode():

    def __init__(self, name, path, depth, node_id, child_idx=-1, parent=None):
        self.name = name                # node name
        self.path = path                # dataset path to this node
        self.depth = depth              # depth of this node
        self.node_id = node_id          # node index in the tree
        self.children = {}              # list of children names
        self.child_idx = child_idx      # child index of its parent
        self.param_list = []            # list of parameter indices for classification at this node
        self.parent = parent            # name of the parent node
        self.codeword = None            # codeword for hierarchical parameterization

    def add_child(self, child):
        self.children[len(self.children)] = child

    def init_codeword(self, cw_size):
        self.codeword = np.zeros([cw_size])

    def set_codeword(self, idx):
        self.codeword[idx] = 1

    def __str__(self):
        attr = 'name={}, node_id={}, depth={}, children={}'.format(
                    self.name, self.node_id, self.depth,
                    ','.join([chd for chd in self.children.values()])
                )
        return  attr

    def copy(self):
        new_node = TreeNode(self.name, self.path, self.depth, self.node_id, self.child_idx, self.parent)
        new_node.children = self.children.copy()
        new_node.codeword = self.codeword
        if self.param_list:
            new_node.param_list = self.param_list.copy()
        return new_node


class Tree():

    def __init__(self, data_root):
        """ Build a tree based on the dataset folder hierarachy.
            data_root: the root directory of the hierarchical dataset
        """
        self.root = TreeNode('root', data_root, 0, 0)
        self.depth = 0                          # the maximum depth of the tree (initialized as 0)
        self.nodes = {'root': self.root}        # list of all the nodes in the tree
        self.nid2name = {0: 'root'}             # a mapping from node index to node name

        # build tree
        self._buildTree(self.root)

        # generate dictionary of internal (non-leaf) nodes (including the root node)
        intnl_nodes = sorted([v for v in self.nodes.values() if len(v.children) > 0], key=lambda x: x.node_id)
        self.intnl_nodes = {i: x.name for i, x in enumerate(intnl_nodes)} # a mapping from intnl_node_id to node name

        # generate dictionary of leaf nodes
        leaf_nodes = sorted([v for v in self.nodes.values() if len(v.children) == 0], key=lambda x: x.node_id)
        self.leaf_nodes = {i: x.name for i, x in enumerate(leaf_nodes)}   # a mapping from leaf_node_id to node name

        # a node is either an internal node or a leaf node
        assert len(self.leaf_nodes) + len(self.intnl_nodes) == len(self.nodes)

        # a mapping from leaf node labels to sublabels for each internal node
        self._gen_sublabels()

    def _buildTree(self, root, depth=0):
        """ Traverse the root directory to build the tree (starting with depth=0).
            root: node to be used as the root
        """

        for chd in os.listdir(root.path):
            chd_path = os.path.join(root.path, chd)

            # if this child is a node (internal/leaf), then add it to the tree
            if os.path.isdir(chd_path):
                assert chd not in self.nodes
                child_idx = len(root.children)
                root.add_child(chd)
                node_id = len(self.nodes)
                child = TreeNode(chd, chd_path, depth + 1, node_id, child_idx, parent=root.name)
                self.nodes[chd] = child
                self.nid2name[node_id] = chd

                # keep traverse its children
                self._buildTree(child, depth + 1)

        self.depth = max(self.depth, depth)

    def _gen_sublabels(self):
        """ Generate sublabels for each internal nodes.
        """
        self.sublabels = -np.ones([len(self.leaf_nodes), len(self.intnl_nodes)])
        name2inid = {v: k for k, v in self.intnl_nodes.items()}
        # generate sublabels for each leaf node class
        for leaf_id, name in self.leaf_nodes.items():
            node = self.nodes.get(name)
            parent = node.parent
            while parent:
                parent_inid = name2inid.get(parent)
                self.sublabels[leaf_id, parent_inid] = node.child_idx
                node = self.nodes.get(parent)
                parent = node.parent

    def show(self, node_name='root', root_depth=-1, max_depth=np.Inf, cls_alias=None):
        """ Display the sub-tree architecture under the specified root within the max_depth.
            node_name: the name of the root node to display
            root_depth: Just for recursive calls; no need to specify root_depth when calling
                this function externally.
            max_depth: the maximum depth to display
            cls_alias: a mapping from the class (leaf node) name to the alias
        """

        root = self.nodes.get(node_name, None)
        if not root:
            raise ValueError('{} is not in the tree'.format(node_name))

        if root_depth == -1:
            if cls_alias is not None and len(root.children) == 0: # leaf node
                print(colored('{} ({})'.format(root.name, cls_alias[root.name]), 'red'))
            else:
                print(root.name)
            root_depth = root.depth
            max_depth = min(self.depth, max_depth)

        if root.depth - root_depth < max_depth:
            for chd in root.children.values():
                child = self.nodes[chd]
                print('--' * (child.depth - root_depth), end='')
                if cls_alias is not None and len(child.children) == 0: # leaf node
                    print(colord('{} ({})'.format(child.name, cls_alias[child.name]), 'red'))
                else:
                    print(child.name)
                self.show(chd, root_depth, max_depth)

    def gen_codewords(self, cw_type='class'):
        """ Generate codewords for all the nodes.
            cw_type: choice = ['class', 'param-td', 'param-bu']
                class: Codewords encoding the classes.
                param-td: Top-down codewords encoding the parameters.
                param-bu: Bottom-up codewords encoding the parameters.
        """
        if cw_type == 'class':
            # leaf nodes
            n_leaf_node = len(self.leaf_nodes)
            for leaf_id, name in self.leaf_nodes.items():
                node = self.nodes.get(name)
                node.init_codeword(n_leaf_node)
                node.set_codeword(leaf_id)

            # internal nodes
            for inid, name in self.intnl_nodes.items():
                node = self.nodes.get(name)
                node.codeword = (self.sublabels[:, inid] >= 0).astype(int)

        elif cw_type == 'param-td' or cw_type == 'param-bu':
            num_nodes = len(self.nodes)
            for node_id, name in self.nid2name.items():
                node = self.nodes.get(name)
                node.init_codeword(num_nodes - 1)
                if node_id > 0:
                    node.set_codeword(node_id - 1)
                    if cw_type=='param-td':
                        # inherent the codeword from the parent node
                        parent_cw = self.nodes.get(node.parent).codeword
                        node.codeword += parent_cw

    def gen_param_lists(self):
        """ Generate parameter list for each internal nodes.
        """
        for name in self.intnl_nodes.values():
            node = self.nodes.get(name)
            for chd in node.children.values():
                param_idx = self.nodes.get(chd).node_id - 1
                node.param_list.append(param_idx)

    def gen_dependence(self):
        """ Generate dependence matrix for internal nodes.
        """
        self.dependence = np.eye(len(self.intnl_nodes))
        name2inid = {v: k for k, v in self.intnl_nodes.items()}
        for inid, name in self.intnl_nodes.items():
            node = self.nodes.get(name)
            parent = node.parent
            while parent:
                parent_inid = name2inid.get(parent)
                self.dependence[inid, parent_inid] = 1
                node = self.nodes.get(parent)
                parent = node.parent

    def get_class(self, node_name='root', verbose=False):
        root = self.nodes.get(node_name, None)
        if not root:
            raise ValueError('{} is not in the tree'.format(node_name))

        def traverse(root):
            for chd in root.children.values():
                child = self.nodes[chd]
                if len(child.children) == 0:
                    class_list.append(child.name)
                traverse(child)

        class_list = []
        if len(root.children) == 0:
            class_list.append(root.name)
        else:
            traverse(root)
        if verbose:
            print(node_name, class_list)
        return class_list

    def get_nodeId(self, node_name=None):

        node = self.nodes.get(node_name, None)
        if not node:
            raise ValueError('{} is not in the tree'.format(node_name))

        return node.node_id

    def get_parent(self, node_name=None):

        node = self.nodes.get(node_name, None)
        if not node:
            raise ValueError('{} is not in the tree'.format(node_name))

        return node.parent


