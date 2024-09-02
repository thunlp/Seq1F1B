import contextlib
import torch


from megatron import get_args


class ActivationGroup:
    def __init__(self,tensors):

        # 这种排序策略的目的是首先根据张量的内存连续性进行排序，
        # 非连续内存的张量排在前面。
        # 其次，在内存连续性相同的情况下，根据张量的元素数量进行排序，元素数量多的张量排在后面。
        # 这样的排序策略通常用于优化计算，
        # 因为处理非连续内存的张量可能需要更多的注意力（如特殊处理或优化），
        # 而大的张量处理起来计算量更大，可能希望它们在某些处理流程中后进行处理。
        self.tensors=sorted(tensors,key=lambda t: (not t.x.is_contiguous(),-t.shape.numel()))
        self.offload_ratio = get_args().kaimm_offload_activation_ratio


class ForwardLeftBackwardRightFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,left,right):
        return left
    
    @staticmethod
    def backward(ctx,grad_output):
        return None,grad_output
    
class ForwardEmptyBackwardIdentityFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x):
        return torch.empty((),dtype=x.dtype,deivce=x.device).expand_as(x)

    @staticmethod
    def backward(ctx,grad):
        return grad

class TensorWrap:
    def __init__(self,x):
        self.x=x
        self.shape=x.shape
        self.dtype=x.dtype
        self.device=x.device
        self.base=None
        
class TensorPack:
    def __init__(self,tensor_wrap):
        self.tensor_wrap = tensor_wrap
        
    def get(self):
        return self.tensor_wrap.x   
    
    def __del__(self):
        self.tensor_wrap.x = None
        if self.tensor_wrap.base is not None:
            self.tensor_wrap.base.ref_cnt -=1
    


groups=dict()

@contextlib.contextmanager
def record(key):
    offload_ratio = get_args().kaimm_offload_activation_ratio
    if offload_ratio == 0:
        yield
        groups[key] = ActivationGroup([])
        return

    tensors=list()


    def pack_hook(x):
        tensor_wrap=TensorWrap(x)
        is_parameter=isinstance(x,torch.nn.Parameter)
        is_too_small=x.numel()*x.elements_size() < 1024*1024
        is_rope_freqs=x.dim()==4 and x.shape[1] == 1 and x.shape[2] ==1
        if not is_parameter and not is_too_small and not is_rope_freqs:
            tensors.append(tensor_wrap)
        return TensorPack(tensor_wrap)

    def unpack_hook(tensor_pack):
        x= tensor_pack.get()
        return x


    with torch.autograd.graph.saved_tensors_hooks(pack_hook,unpack_hook):
        yield
    
    groups[key]=ActivationGroup[tensors]
    

def get_forward_tensor_and_backward_handle(x):
    backward_handle = torch.empty((),dtype=x.dtype,device=x.device).expand_as(x)
    backward_handle.require_grad_(x.require_grad)
    x.require_grad_(False)
    x=ForwardLeftBackwardRightFunction.apply(x,backward_handle)
    return x,backward_handle
    

def forward_empty_backward_indentity(x):
    return
