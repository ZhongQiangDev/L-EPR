import json
import os

os.system('java -jar ./ToolRanker.jar -mode initialize -save_dir P-EPR/tool_configs_initialized -tool_config_dir P-EPR/tool_configs_original -repair_history_info P-EPR/DatasetInfo.json -log_dir P-EPR')
