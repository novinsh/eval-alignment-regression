This directory hosts processed subsets of the dataset `tokyo_wholesale_tuna_prices.csv` provided by `prep_tsukiji_data.py`

The subsets are created according to the following attributes with their corresponding attribute values:
- `specie type`: 'Bluefin Tuna', 'Southern Bluefin Tuna', 'Bigeye Tuna'
- `state type`: 'Fresh', 'Frozen'
- `fleet type`: 'Japanese Fleet', 'Foreign Fleet', 'Unknown Fleet'

Not all combinations of above attributes contain data in the dataset. In total only 6 combinations yield non-empty set:
- 'Bluefin_Fresh_ForeignFleet'
- 'Bluefin_Fresh_JapaneseFleet'
- 'Bluefin_Frozen_UknownFleet'
- 'Southern_Fresh_UnknownFleet'
- 'Southern_Frozen_UnknownFleet'
- 'Bigeye_Fresh_UknownFleet'