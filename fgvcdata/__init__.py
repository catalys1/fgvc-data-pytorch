'''A common interface to FGVC datasets.

Currently supported datasets are
- CUB Birds
- CUB Birds with expert labels
- NA Birds
- Stanford Cars
- Stanford Dogs
- Oxford Flowers
- Oxford FGVC Aircraft
- Tsinghua Dogs

Datasets are constructed and used following the pytorch
data.utils.data.Dataset paradigm, and have the signature

fgvcdata.Dataset(root='path/to/data/'[,transform[,target_transform[,train]]])

`root` is the path to the base folder for the dataset. Additionally, `root` can
end in `/train` or `/test`, to indicate whether to use train or test data --
even if the root folder does not contain `train` or `test` subfolders.

The use of training or test data can also be specified through the use of the
`train` flag (the path extension on `root` takes precedence).

`transform` and `target_transform` are optional callables that preprocess data
and targets respectively; for instance, torchvision.transforms.
'''
from . import birds
from .birds import *
from . import cars
from .cars import *
from . import dogs
from .dogs import *
from . import aircraft
from .aircraft import *
from . import flowers
from .flowers import *
from . import icub
from .icub import *
from .fungi import *


IMAGENET_STATS = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

datasets = []
for f in [birds,icub,cars,dogs,aircraft,flowers,fungi]:
    datasets += f.__all__
