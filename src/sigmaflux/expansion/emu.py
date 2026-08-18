"""

"""
from collections.abc import Iterable
import cobra

from sigmaflux.expansion.utils import can_make_met


def _produces_from_nothing(rxn, metabolite):
    """Whether rxn produces metabolite "from nothing" (no educt on the producing side), marking it as a network input (boundary).

    Args:
        rxn: reaction to check.
        metabolite: metabolite to check for boundary production of.
    """
    # forward production: metabolite is a product, educts are the reactants
    if rxn.upper_bound > 0 and metabolite in rxn.products and len(rxn.reactants) == 0:
        return True
    # reverse production: metabolite is a reactant, educts are the products
    if rxn.lower_bound < 0 and metabolite in rxn.reactants and len(rxn.products) == 0:
        return True
    return False


class EMU(object):
    """An Elementary Metabolite Unit: a metabolite plus a set of 1-based carbon positions it covers."""

    def __init__(self, metabolite, mid):

        """Args:
            metabolite: cobra.Metabolite this EMU covers part of.
            mid: 1-based carbon positions this EMU covers, e.g. [1,2,3,4].
        """
        if not isinstance(metabolite, cobra.Metabolite):
            raise TypeError(f'Input {metabolite} is of type {type(metabolite)} must be cobra.Metabolite')

        # Boundary status is a property of the stoichiometry and must be read
        # from the live (model-attached) metabolite BEFORE the copy below
        # detaches it (metabolite.reactions would then be empty).  An EMU is a
        # boundary (network input) iff an atom-mapped reaction produces its
        # metabolite "from nothing" -- the production has no upstream educt.
        self._is_boundary = any(
            'atom_mapping' in rxn.annotation
            and can_make_met(rxn, metabolite)
            and _produces_from_nothing(rxn, metabolite)
            for rxn in metabolite.reactions
        )

        self._metabolite = metabolite.copy()
        self.mid = mid

        # Get symmetries from Metabolite.annotation
        symetries = metabolite.annotation.get('symmetries', [])
        self.symmetries = [parse_symetry(s) for s in symetries]

    @property
    def size(self) -> int:
        return len(self.mid)

    # TODO THIS IS A BUGGY PROPERTY NEEDS TO BE RECONCILED 
    # ID <> METABOLITE are used interchangably
    @property
    def id(self) -> str:
        return self._metabolite.id
    
    @property
    def metabolite(self) -> str:
        return self._metabolite.id

    @property
    def is_boundary(self) -> bool:
        return self._is_boundary

    def __eq__(self, other):
        if isinstance(other, EMU):
            return (self.id == other.id) & (self.mid == other.mid)
        elif isinstance(other, OriginSet) and len(other) == 1:
            return (self.id == other._values[0].id) & \
                   (self.mid == other._values[0].mid)
        return False

    def __hash__(self):
        return hash(self.__repr__())

    def __repr__(self):
        return str(self.id) + '_' + ''.join([str(i) for i in self.mid])


class OriginSet(object):
    def __init__(self, value):
        """Args:
            value: a single EMU, or an iterable of EMUs, to wrap as one origin set.
        """
        self.current = -1

        if isinstance(value, EMU):
            self._values = [value]
        elif isinstance(value, Iterable):
            self._values = [*value]
        else:
            raise TypeError(f'Input {value} is of type {type(value)} must be Iterable or EMU')

    @property
    def size(self):
        return sum([v.size for v in self._values])

    @property
    def is_boundary(self) -> bool:
        return any(v._is_boundary for v in self._values) or len(self._values) > 1

    def __len__(self):
        return len(self._values)

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        if self.current < len(self._values):
            return self._values[self.current]
        raise StopIteration

    def __repr__(self):
        return ' + '.join([v.__repr__() for v in self._values])

    def __hash__(self):
        return hash(self.__repr__())

    def __bool__(self):
        if len(self._values) > 0:
            return True
        else:
            return False

    def __contains__(self, item):
        if isinstance(item, Iterable):
            return any([i in self._values for i in item])
        elif isinstance(item, OriginSet):
            return any([i in self._values for i in item._values])
        else:
            return item in self._values

    def __eq__(self, other):
        if isinstance(other, EMU) and len(self._values) == 1:
            return self._values[0] == other
        elif not isinstance(other, OriginSet):
            return False
        else:
            return self._values == other._values

    def __add__(self, other):
        return OriginSet(self._values + other._values)


def get_metabolite(item):
    if isinstance(item, OriginSet) and len(item) == 1:
        return [item._values[0].id]
    elif isinstance(item, EMU):
        return [item.id]
    elif isinstance(item, OriginSet) and len(item) > 1:
        return [i.id for i in item._values]
    else:
        raise ValueError('TODO')


def parse_symetry(s):
    """Parse a symmetry string like 'ABC.ACB' into a tuple ('ABC', 'ACB')."""
    parts = s.split('.')
    if len(parts) != 2:
        raise ValueError(f'Error in symetry string {s}')
    return (parts[0], parts[1])