# Import the required modules
from anomalib.data import MVTec
from anomalib.models import Patchcore
from anomalib.engine import Engine

# Initialize the datamodule, model and engine
datamodule = MVTec()
model = Patchcore()
engine = Engine()

# Train the model
engine.fit(datamodule=datamodule, model=model)