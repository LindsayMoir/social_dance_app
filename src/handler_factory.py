"""
Handler factory for creating consistent handler configurations across scrapers.

This module consolidates handler initialization logic that was previously
scattered and duplicated across multiple scraper subclasses. By providing
a factory method, we ensure:

1. Consistent handler configuration across all scrapers
2. Single source of truth for handler dependencies
3. Reduced code duplication (~40-50 lines per subclass)
4. Conditional creation of expensive resources (DB, PDF handlers)
5. Easier testing and mocking of handler configurations

Usage:
    from handler_factory import HandlerFactory

    # Create minimal set of handlers
    handlers = HandlerFactory.create_standard_handlers(
        config=config_dict,
        logger=logger_instance,
    )

    # Create with database and authentication
    handlers = HandlerFactory.create_standard_handlers(
        config=config_dict,
        logger=logger_instance,
        include_database=True,
        include_auth=True,
    )
"""

import logging
from typing import Dict, Optional, Any

from browser_utils import PlaywrightManager
from text_utils import TextExtractor
from auth_manager import AuthenticationManager
from resilience import RetryManager, CircuitBreaker
from url_nav import URLNavigator
from pdf_utils import PDFExtractor
from llm import LLMHandler
from db_utils import DBWriter


class HandlerFactory:
    """
    Factory for creating and configuring scraper handler objects.

    Consolidates the creation of handlers with consistent configuration,
    reducing duplication across BaseScraper subclasses.

    Handler Groups:
    - Core Handlers: Always created (browser, text extraction, resilience)
    - Optional Handlers: Created based on flags (database, PDF, authentication)
    - High-Cost Handlers: Only instantiated when needed (LLMHandler, DatabaseHandler)
    """

    @staticmethod
    def create_standard_handlers(
        config: Dict[str, Any],
        logger: logging.Logger,
        include_database: bool = False,
        include_pdf: bool = False,
        include_auth: bool = True,
    ) -> Dict[str, Any]:
        """
        Create and configure standard handler objects for scraper initialization.

        This factory method eliminates redundant handler initialization code
        that was previously duplicated across fb_v2, ebs_v2, rd_ext_v2,
        read_pdfs_v2, and other scraper subclasses.

        Core Handlers (Always Created):
        - browser_manager: PlaywrightManager for Playwright automation
        - text_extractor: TextExtractor for HTML parsing and text extraction
        - retry_manager: RetryManager for exponential backoff retry logic
        - circuit_breaker: CircuitBreaker for fault tolerance
        - url_navigator: URLNavigator for URL tracking and deduplication

        Optional Handlers (Created Based on Flags):
        - auth_manager: AuthenticationManager for sites requiring login
        - pdf_extractor: PDFExtractor for PDF-based scrapers
        - llm_handler: LLMHandler for LLM-based event extraction
        - db_writer: DBWriter for database operations (requires llm_handler)

        Args:
            config (Dict[str, Any]): Configuration dictionary loaded from YAML.
                                    Must contain all necessary config values
                                    for handler initialization.
            logger (logging.Logger): Logger instance to be used by all handlers.
                                    Ensures consistent logging across components.
            include_database (bool, optional): If True, creates LLMHandler and
                                             DBWriter for database operations.
                                             Default: False
            include_pdf (bool, optional): If True, creates PDFExtractor for
                                         PDF processing. Default: False
            include_auth (bool, optional): If True, creates AuthenticationManager
                                          for handling login operations.
                                          Default: True (most scrapers need auth)

        Returns:
            Dict[str, Any]: Dictionary containing initialized handler instances.
                           Keys include:
                           - 'browser_manager': PlaywrightManager instance
                           - 'text_extractor': TextExtractor instance
                           - 'retry_manager': RetryManager instance
                           - 'circuit_breaker': CircuitBreaker instance
                           - 'url_navigator': URLNavigator instance
                           - 'auth_manager': AuthenticationManager (if include_auth=True)
                           - 'pdf_extractor': PDFExtractor (if include_pdf=True)
                           - 'llm_handler': LLMHandler (if include_database=True)
                           - 'db_writer': DBWriter (if include_database=True)

        Example:
            Basic scraper with just browser automation:
            >>> handlers = HandlerFactory.create_standard_handlers(
            ...     config=config,
            ...     logger=logger,
            ...     include_database=False,
            ... )
            >>> browser_mgr = handlers['browser_manager']
            >>> text_ext = handlers['text_extractor']

            Full-featured scraper with database and authentication:
            >>> handlers = HandlerFactory.create_standard_handlers(
            ...     config=config,
            ...     logger=logger,
            ...     include_database=True,
            ...     include_auth=True,
            ... )
            >>> db_writer = handlers['db_writer']
            >>> auth_mgr = handlers['auth_manager']

        Raises:
            Exception: If handler initialization fails (e.g., missing config keys,
                      database connection issues). The specific exception depends
                      on which handler failed.

        Note:
            Order of handler creation matters:
            1. Core handlers (independent, no dependencies)
            2. Optional utilities (auth, pdf - still independent)
            3. Database handlers (depend on config and logger, costly to create)
            This order ensures proper resource management and error handling.
        """
        handlers: Dict[str, Any] = {}

        # Phase 1: Create core handlers (always needed, no dependencies)
        logger.debug("Creating core handler instances...")
        handlers["browser_manager"] = PlaywrightManager(config)
        handlers["text_extractor"] = TextExtractor(logger)
        handlers["retry_manager"] = RetryManager(logger=logger)
        handlers["circuit_breaker"] = CircuitBreaker(logger=logger)
        handlers["url_navigator"] = URLNavigator(logger)
        logger.debug("Core handlers created successfully")

        # Phase 2: Create optional authentication handler
        if include_auth:
            logger.debug("Creating AuthenticationManager...")
            handlers["auth_manager"] = AuthenticationManager(logger)
            logger.debug("AuthenticationManager created successfully")

        # Phase 3: Create optional PDF handler
        if include_pdf:
            logger.debug("Creating PDFExtractor...")
            handlers["pdf_extractor"] = PDFExtractor(logger=logger)
            logger.debug("PDFExtractor created successfully")

        # Phase 4: Create optional database handlers (high-cost operations)
        if include_database:
            logger.debug("Creating LLMHandler and database components...")
            try:
                # LLMHandler creates DatabaseHandler internally
                handlers["llm_handler"] = LLMHandler()

                # Create DBWriter wrapper for database operations
                handlers["db_writer"] = DBWriter(
                    handlers["llm_handler"].db_handler,
                    logger
                )
                logger.debug("Database handlers created successfully")
            except Exception as e:
                logger.error(f"Failed to create database handlers: {e}")
                raise

        logger.info(
            f"Handler factory created {len(handlers)} handlers "
            f"(database={include_database}, pdf={include_pdf}, auth={include_auth})"
        )

        return handlers

    @staticmethod
    def create_web_scraper_handlers(
        config: Dict[str, Any],
        logger: logging.Logger,
    ) -> Dict[str, Any]:
        """
        Create handlers optimized for web scraping (authentication + database).

        Convenience method for creating handlers with the most common configuration
        for web scrapers like FacebookScraperV2, EventbriteScraperV2, and
        ReadExtractV2 that need both authentication and database writing.

        Args:
            config (Dict[str, Any]): Configuration dictionary from YAML
            logger (logging.Logger): Logger instance

        Returns:
            Dict[str, Any]: Handlers dictionary with all features enabled

        Example:
            >>> handlers = HandlerFactory.create_web_scraper_handlers(config, logger)
            >>> # Equivalent to:
            >>> handlers = HandlerFactory.create_standard_handlers(
            ...     config, logger,
            ...     include_database=True,
            ...     include_auth=True,
            ... )
        """
        return HandlerFactory.create_standard_handlers(
            config=config,
            logger=logger,
            include_database=True,
            include_auth=True,
            include_pdf=False,
        )

    @staticmethod
    def create_pdf_scraper_handlers(
        config: Dict[str, Any],
        logger: logging.Logger,
    ) -> Dict[str, Any]:
        """
        Create handlers optimized for PDF processing (database + PDF extraction).

        Convenience method for creating handlers with PDF extraction capability
        for scrapers like ReadPDFsV2 that need PDF processing and database writing.

        Args:
            config (Dict[str, Any]): Configuration dictionary from YAML
            logger (logging.Logger): Logger instance

        Returns:
            Dict[str, Any]: Handlers dictionary with PDF and database features

        Example:
            >>> handlers = HandlerFactory.create_pdf_scraper_handlers(config, logger)
            >>> # Equivalent to:
            >>> handlers = HandlerFactory.create_standard_handlers(
            ...     config, logger,
            ...     include_database=True,
            ...     include_auth=False,
            ...     include_pdf=True,
            ... )
        """
        return HandlerFactory.create_standard_handlers(
            config=config,
            logger=logger,
            include_database=True,
            include_auth=False,
            include_pdf=True,
        )

    @staticmethod
    def create_crawler_handlers(
        config: Dict[str, Any],
        logger: logging.Logger,
    ) -> Dict[str, Any]:
        """
        Create handlers for read-only web crawling (minimal configuration).

        Convenience method for creating a minimal set of handlers for
        read-only web crawlers that don't need authentication or database.

        Args:
            config (Dict[str, Any]): Configuration dictionary from YAML
            logger (logging.Logger): Logger instance

        Returns:
            Dict[str, Any]: Handlers dictionary with core features only

        Example:
            >>> handlers = HandlerFactory.create_crawler_handlers(config, logger)
            >>> # Equivalent to:
            >>> handlers = HandlerFactory.create_standard_handlers(
            ...     config, logger,
            ...     include_database=False,
            ...     include_auth=False,
            ...     include_pdf=False,
            ... )
        """
        return HandlerFactory.create_standard_handlers(
            config=config,
            logger=logger,
            include_database=False,
            include_auth=False,
            include_pdf=False,
        )
