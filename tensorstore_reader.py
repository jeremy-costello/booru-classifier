import random
import tensorstore as ts    


class CustomTensorStoreDataLoader:
    def __init__(self, length, batch_size, shuffle):
        self.length = length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.reset_indices()
    
    async def init_tensorstores(self, tensorstore_inputs_dict):
        self.tensorstore_dict = dict()
        for key, value in tensorstore_inputs_dict.items():
            self.tensorstore_dict[key] = await self.open_tensorstore(
                value["tensorstore_file"],
                value["cache_limit"]
            )
    
    def reset_indices(self):
        self.remaining_indices = list(range(self.length))
        if self.shuffle:
            random.shuffle(self.remaining_indices)
    
    @staticmethod
    async def open_tensorstore(tensorstore_file, cache_limit):
        return await ts.open({
            "driver": "zarr",
            "kvstore": {
                "driver": "file",
                "path": tensorstore_file
            },
            "context": {
                "cache_pool": {
                    "total_bytes_limit": cache_limit
                }
            },
            "recheck_cached_data": "open"
        })
    
    def get_indices(self):
        indices = self.remaining_indices[-self.batch_size:]
        self.remaining_indices = self.remaining_indices[:-self.batch_size]
        return indices
    
    async def get_batch(self):
        indices = self.get_indices()
        return await self.batch_logic(indices)
    
    async def batch_logic(self, indices):
        raise NotImplementedError()
