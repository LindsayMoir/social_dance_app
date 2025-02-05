# base_handler.py
import yaml
import logging


class BaseHandler:
    _config = None
    _logging_configured = False
    _db_handler = None  # Shared DB handler

    def __init__(self, config_path="config/config.yaml"):
        # Load configuration once globally
        if BaseHandler._config is None:
            with open(config_path, "r") as file:
                BaseHandler._config = yaml.safe_load(file)
        self.config = BaseHandler._config

        # Configure logging once globally
        if not BaseHandler._logging_configured:
            logging.basicConfig(
                filename=self.config['logging']['log_file'],
                filemode='w',
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            BaseHandler._logging_configured = True
            logging.info("Global logging configured.")

        # Lazy import to break circular dependency for DatBaseHandler
        if BaseHandler._db_handler is None:
            from db import DatabaseHandler
            BaseHandler._db_handler = DatabaseHandler(self.config)
        self.db_handler = BaseHandler._db_handler

        logging.info(f"{self.__class__.__name__} initialized.")
