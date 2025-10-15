import logging
import time

class LogFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",    # 青色
        "INFO": "\033[32m",     # 绿色
        "WARNING": "\033[33m",  # 黄色
        "ERROR": "\033[31m",    # 红色
        "CRITICAL": "\033[41m", # 红底
    }
    RESET = "\033[0m"

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, "")
        reset = self.RESET if log_color else ""
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        return f"{log_color}[{time_str}] [{record.levelname}] {record.getMessage()}{reset}"

# 创建 logger
log = logging.getLogger("project_logger")
log.setLevel(logging.INFO)

# 检查是否已有 console handler，避免重复添加
if not log.hasHandlers():
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(LogFormatter())
    log.addHandler(_console_handler)
