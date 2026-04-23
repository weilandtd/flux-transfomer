"""

"""
from collections.abc import Iterable
import cobra


class EMU(object):
    """
    Class method for EMU.
    """

    def __init__(self, metabolite, mid, ix=None):

        """
        :param metabolite: cobra.Metabolite
        :param mid: [1,2,3,4] (EMU)
        """
        if not isinstance(metabolite, cobra.Metabolite):
            raise TypeError(f'Input {metabolite} is of type {type(metabolite)} must be cobra.Metabolite')
        self._metabolite = metabolite.copy()
        self.mid = mid

        # Get symmetries from Metabolite.annotation
        symetries = metabolite.annotation.get('symmetries', [])
        self.symmetries = [parse_symetry(s) for s in symetries]

        if ix is None:
            self.ix = [ i for i in range(len(mid))]
        else:
            self.ix = ix

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

    def __eq__(self, other):
        if isinstance(other, EMU):
            #return (self.id == other.id) & (self.mid == other.mid)
            return (self.id == other.id) & (self.mid == other.mid) #& (self.ix == other.ix)
        elif isinstance(other, OriginSet) and len(other) == 1:
            # return (self.id == other._values[0].id) \
            #        & (self.mid == other._values[0].mid)
            return (self.id == other._values[0].id) & \
                   (self.mid == other._values[0].mid) #& \
                   #(self.ix == other._values[0].ix)  
            
        return False 

    def __hash__(self):
        return hash(self.__repr__())

    def __repr__(self):
        return str(self.id) + \
        '_' + ''.join([str(i) for i in self.mid]) #+ \
        #'_' + ''.join([str(i) for i in self.ix])


class OriginSet(object):
    def __init__(self, value):
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
    """
    Parse a symetry string like 'ABC,ACB' into a tuple ('ABC', 'ACB')
    :param s: str
    :return: tuple
    """
    parts = s.split('.')
    if len(parts) != 2:
        raise ValueError(f'Error in symetry string {s}')
    return (parts[0], parts[1])