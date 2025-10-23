import upstox_client, ta, pandas as pd
print("upstox_client OK", getattr(upstox_client, "__version__", ""))
print("ta OK", ta.__version__)
print("pandas OK", pd.__version__)