
__author__ = 'Robbert Harms'
__date__ = "2015-03-26"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Tree:

    def __init__(self, data=None, tag=None, children=None, parent=None):
        """Create a new Tree.

        In this tree, every node is a tree object as well. The tree is implemented as a linked list.
        Each node has a reference to its children and to its parent node.

        Args:
            data : The data object
            tag (str): The tag used for displaying this node
            children (list of Tree): The list of children to this node
            parent (Tree): The parent tree node.

        Attributes:
            data : The data object
            tag (str): The tag used for displaying this node
            children (list of Tree): The list of children to this node
            parent (Tree): The parent tree node.
        """
        self.data = data
        self.tag = tag or ""
        self.children = children or []
        self.parent = parent

    @property
    def leaves(self):
        """Get all the leaves under this tree.

        Returns:
            list: A list of all leaves under this tree.
        """
        leaves = []
        if not self.children:
            leaves.append(self)
        else:
            for child in self.children:
                leaves.extend(child.leaves)
        return leaves

    @property
    def internal_nodes(self):
        """Get all the non-leaves under this tree (the internal nodes).

        Returns:
            list: A list of all non-leaves under this tree.
        """
        internal_nodes = []
        if self.children:
            internal_nodes.append(self)
            for child in self.children:
                internal_nodes.extend(child.internal_nodes)
        return internal_nodes

    def __str__(self, level=0):
        ret = "\t"*level + self.tag + "\n"
        for child in self.children:
            ret += child.__str__(level+1)
        return ret


class CompartmentModelTree(Tree):

    def __init__(self, model_lists):
        """Builds a multi modal multi compartment model from the given model tree.

        Valid model trees abides this grammar:

        tree     ::= model | '(' tree ')' | '(' tree ',' operator ')'
        model    ::= CompartmentModel ['(' nickname ')']
        operator ::= '*' | '/' | '+' | '-'

        This means that one can build complex models consisting of compartment models combined using basic
        math operators.

        Args:
            model_lists (list of mdt.models.compartments.CompartmentModel): The model tree list
        """
        super().__init__()
        self._init_tree(model_lists)

    def get_compartment_models(self):
        """Get the compartment models that are part of this tree.

        This basically just returns the leaves of the tree.

        Returns:
            list of mdt.models.compartments.CompartmentModel: the compartments in this tree
        """
        return [n.data for n in self.leaves]

    def _init_tree(self, listing):
        if isinstance(listing, (list, tuple)):
            if len(listing) == 1:
                self.data = listing[0]
                self.tag = listing[0].name
            else:
                operator = None
                for node in listing:
                    if isinstance(node, str):
                        if operator is not None:
                            raise ValueError('Double operator in model listing.')
                        operator = node
                    else:
                        nn = CompartmentModelTree(node)
                        nn.parent = self
                        self.children.append(nn)

                if operator is None:
                    raise ValueError('No operator in model listing.')

                self.data = operator
                self.tag = operator
        else:
            self.data = listing
            self.tag = listing.name

    def __str__(self, level=0):
        if isinstance(self.data, str):
            operator = ' ' + self.data + ' '
            return '(' + "\n" + "\t" * (level + 1) + \
                   operator.join([child.__str__(level + 1) for child in self.children]) + \
                   "\n" + "\t" * level + ')'
        else:
            return self.data.name + '(' + ', '.join([p.name for p in self.data.get_parameters()]) + ')'
