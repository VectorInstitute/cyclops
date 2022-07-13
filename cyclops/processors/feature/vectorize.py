from typing import Dict, List, Union, Optional

import numpy as np


class Vectorized:
    """Vectorized data.
    
    Attributes
    ----------
    indexes: list of numpy.ndarray
        AAA.
    """
    def __init__(
        self,
        data: np.ndarray,
        indexes: Optional[List[Union[List, np.ndarray]]] = None,
    ) -> None:
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy.ndarray.")
        
        if indexes is not None:
            if not len(data.shape) == len(indexes):
                raise ValueError(
                    "Number of array axes and the number of indexes do not match."
                )

            for i, index in enumerate(indexes):
                if not isinstance(index, list) and not isinstance(index, np.ndarray):
                    raise ValueError("Indexes must be a list of list or numpy.ndarray.")

                index = np.array(index)
                    
                if len(index) != data.shape[i]:
                    raise ValueError(
                        (f"Axis {i} index has {len(index)} elements,",
                         "but the axis itself has length {data.shape[i]}.")
                    )
                
                if len(np.unique(index)) != len(index):
                    raise ValueError(
                        "Each index must have no duplicate values to uniquely identify."
                    )
                
                indexes[i] = index
            
            index_maps = [{val: i for i, val in enumerate(index)} for index in indexes]
            index_maps_inv = [{v: k for k, v in index.items()} for index in index_maps]
        else:
            index_maps = None
            index_maps_inv = None
            
        self.data: np.ndarray = data
        self.indexes: List[np.ndarray] = indexes
        self.index_maps: Dict[str, int] = index_maps
        self.index_maps_inv: Dict[int, str] = index_maps_inv
        
    

    def get_data(self) -> np.ndarray:
        return self.data
    
    def get_by_index(self, *indexes) -> np.ndarray:
        """Get data by indexing each axis.
        
        Parameters
        ----------
        *indexes
            E.g., (None, [1, 2, 3], None, [20])
        
        Returns
        -------
        numpy.ndarray
            Indexed data.

        """
        data = self.data
        print("data.shape", data.shape)
        new_indexes = None if self.indexes is None else []
        for i, index in enumerate(indexes):
            if index is None:
                if self.indexes is not None:
                    new_indexes.append(self.indexes[i])
                continue
            
            # Reshape idx to have same number of dimensions as the data
            print("index - A", index)
            index = np.array(index)
            print("index - B", index)
            shape = [1] * len(data.shape)
            shape[i] = len(index)
            data = np.take_along_axis(data, index.reshape(shape), axis=i)
            print("data.shape", data.shape)
            
            if self.indexes is not None:
                print("index", index)
                new_indexes.append([self.index_maps_inv[i][val] for val in index])
        
        if self.indexes is None:
            return Vectorized(data)
        
        return Vectorized(data, indexes=new_indexes)
        
    def get_by_value(self, *indexes) -> np.ndarray:
        """Get data by indexing using the indexes values for each axis.
        
        Parameters
        ----------
        *indexes
            Length of indexes must equal the number of axes in the data.
            E.g., for array (2, 3, 5), we might specify get_by_value(None, ["A", "B"], None)
            A None value will take all values along an axis
        
        Returns
        -------
        numpy.ndarray
            Indexed data.

        """
        if len(indexes) != len(self.data.shape):
            raise ValueError(
                "Must have the same number of parameters as axes in the data."
            )
        
        indexes = list(indexes)
        for i, index in enumerate(indexes):
            if index is None:
                continue
            
            # Map values to indices
            indexes[i] = [self.index_maps[i][val] for val in index]
        
        return self.get_by_index(*indexes)
        

    def standardize(self, axis: int) -> None:
        if axis < 0 or axis >= len(data.shape):
            raise ValueError("Axis out of bounds.")
        
        self.data = (self.data - self.data.mean(axis=axis))/(self.data.std(axis=axis))