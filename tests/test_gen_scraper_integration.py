#!/usr/bin/env python3
"""
Comprehensive integration tests for GeneralScraper (src/gen_scraper.py).

Tests cover:
- Parallel and sequential pipeline execution
- Performance metrics tracking
- Hash caching and deduplication
- Error recovery and fault tolerance
- Resource management
- Logging and statistics

Mock strategy:
- Mock external components (ReadExtractV2, ReadPDFsV2, LLMHandler)
- Simulate real event data
- Test deduplication logic thoroughly
- Verify performance metrics collection
"""

import asyncio
import logging
import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime
import pandas as pd

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestGeneralScraperIntegration:
    """Integration tests for GeneralScraper parallel and sequential pipelines."""

    @pytest.fixture
    def mock_gen_scraper(self):
        """
        Create a mocked GeneralScraper instance for testing.

        Mocks:
        - LLMHandler initialization
        - ReadExtractV2 initialization
        - ReadPDFsV2 initialization
        - EventSpiderV2 (unavailable)
        - Database writer
        """
        import sys
        sys.path.insert(0, 'src')

        with patch('llm.LLMHandler') as mock_llm, \
             patch('rd_ext_v2.ReadExtractV2') as mock_read_extract, \
             patch('read_pdfs_v2.ReadPDFsV2') as mock_read_pdfs, \
             patch('gen_scraper.yaml.safe_load', return_value={}):

            from gen_scraper import GeneralScraper

            # Create instance normally
            scraper = GeneralScraper()

            # Replace the real attributes with mocks to avoid external dependencies
            scraper.logger = MagicMock()
            scraper.circuit_breaker = MagicMock()
            scraper.db_writer = None
            scraper.llm_handler = mock_llm.return_value
            scraper.read_extract = mock_read_extract.return_value
            scraper.read_pdfs = mock_read_pdfs.return_value

            return scraper

    def sample_events_calendar(self) -> list:
        """Generate sample calendar events for testing."""
        return [
            {
                'Name_of_the_Event': 'Salsa Night at Coda',
                'Start_Date': '2025-10-25',
                'URL': 'https://gotothecoda.com/events/1',
                'location': 'San Francisco'
            },
            {
                'Name_of_the_Event': 'Bachata Basics',
                'Start_Date': '2025-10-26',
                'URL': 'https://gotothecoda.com/events/2',
                'location': 'San Francisco'
            },
            {
                'Name_of_the_Event': 'Swing Dance',
                'Start_Date': '2025-10-27',
                'URL': 'https://example.com/events/3',
                'location': 'Oakland'
            }
        ]

    def sample_events_pdf(self) -> list:
        """Generate sample PDF-extracted events for testing."""
        return [
            {
                'Name_of_the_Event': 'Jazz Concert',
                'Start_Date': '2025-10-25',
                'URL': 'https://concerts.com/pdf1',
                'location': 'Berkeley'
            },
            {
                'Name_of_the_Event': 'Salsa Night at Coda',  # Duplicate from calendar
                'Start_Date': '2025-10-25',
                'URL': 'https://gotothecoda.com/events/1',
                'location': 'San Francisco'
            }
        ]

    @pytest.mark.asyncio
    async def test_parallel_pipeline_basic_execution(self, mock_gen_scraper):
        """Test basic parallel pipeline execution with mocked extractors."""
        # Mock the extraction methods
        calendar_data = self.sample_events_calendar()
        pdf_data = self.sample_events_pdf()

        mock_gen_scraper.extract_from_calendars_async = AsyncMock(
            return_value=pd.DataFrame(calendar_data)
        )
        mock_gen_scraper.extract_from_pdfs_async = AsyncMock(
            return_value=pd.DataFrame(pdf_data)
        )
        mock_gen_scraper.extract_from_websites_async = AsyncMock(
            return_value=pd.DataFrame()
        )

        # Run parallel pipeline
        result = await mock_gen_scraper.run_pipeline_parallel()

        # Verify results
        assert isinstance(result, pd.DataFrame), "Result should be DataFrame"
        assert len(result) > 0, "Should extract events"
        mock_gen_scraper.extract_from_calendars_async.assert_called_once()
        mock_gen_scraper.extract_from_pdfs_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_sequential_pipeline_basic_execution(self, mock_gen_scraper):
        """Test sequential pipeline execution."""
        calendar_data = self.sample_events_calendar()
        pdf_data = self.sample_events_pdf()

        mock_gen_scraper.extract_from_calendars_async = AsyncMock(
            return_value=pd.DataFrame(calendar_data)
        )
        mock_gen_scraper.extract_from_pdfs_async = AsyncMock(
            return_value=pd.DataFrame(pdf_data)
        )
        mock_gen_scraper.extract_from_websites_async = AsyncMock(
            return_value=pd.DataFrame()
        )

        # Run sequential pipeline
        result = await mock_gen_scraper.run_pipeline_sequential()

        # Verify results
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        mock_gen_scraper.extract_from_calendars_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_deduplication_across_sources(self, mock_gen_scraper):
        """Test that deduplication correctly identifies duplicates across sources."""
        calendar_data = self.sample_events_calendar()
        pdf_data = self.sample_events_pdf()

        mock_gen_scraper.extract_from_calendars_async = AsyncMock(
            return_value=pd.DataFrame(calendar_data)
        )
        mock_gen_scraper.extract_from_pdfs_async = AsyncMock(
            return_value=pd.DataFrame(pdf_data)
        )
        mock_gen_scraper.extract_from_websites_async = AsyncMock(
            return_value=pd.DataFrame()
        )

        result = await mock_gen_scraper.run_pipeline_parallel()

        # Should have 4 unique events (3 from calendar + 1 new from pdf)
        # "Salsa Night at Coda" is duplicate
        assert len(result) == 4, f"Expected 4 unique events, got {len(result)}"
        assert mock_gen_scraper.stats['duplicates_removed'] == 1

    def test_hash_caching(self, mock_gen_scraper):
        """Test that hash caching improves performance."""
        event = {
            'Name_of_the_Event': 'Test Event',
            'Start_Date': '2025-10-25',
            'URL': 'https://example.com/1'
        }

        # First call - cache miss
        hash1 = mock_gen_scraper._create_event_hash(event)
        assert mock_gen_scraper.performance_metrics['cache_misses'] >= 1

        # Second call - should hit cache
        hash2 = mock_gen_scraper._create_event_hash(event)
        assert hash1 == hash2, "Same event should produce same hash"
        assert mock_gen_scraper.performance_metrics['cache_hits'] >= 1

    def test_hash_cache_limit(self, mock_gen_scraper):
        """Test that hash cache respects size limit."""
        # Set low limit for testing
        mock_gen_scraper.hash_cache_limit = 5

        # Add events until cache is full
        for i in range(10):
            event = {
                'Name_of_the_Event': f'Event {i}',
                'Start_Date': '2025-10-25',
                'URL': f'https://example.com/{i}'
            }
            mock_gen_scraper._create_event_hash(event)

        # Cache should not exceed limit
        assert len(mock_gen_scraper.event_hash_cache) <= 5

    def test_deduplication_tracking(self, mock_gen_scraper):
        """Test that deduplication properly tracks statistics."""
        events = [
            {'Name_of_the_Event': 'Event A', 'Start_Date': '2025-10-25', 'URL': 'url1'},
            {'Name_of_the_Event': 'Event B', 'Start_Date': '2025-10-26', 'URL': 'url2'},
            {'Name_of_the_Event': 'Event A', 'Start_Date': '2025-10-25', 'URL': 'url1'},  # Duplicate
        ]

        unique = mock_gen_scraper.deduplicate_events(events, source='test')

        assert len(unique) == 2, "Should have 2 unique events"
        assert mock_gen_scraper.stats['duplicates_removed'] == 1
        assert mock_gen_scraper.stats['duplicates_kept'] == 2

    def test_performance_metrics_initialization(self, mock_gen_scraper):
        """Test that performance metrics are initialized correctly."""
        assert 'hash_computations' in mock_gen_scraper.performance_metrics
        assert 'cache_hits' in mock_gen_scraper.performance_metrics
        assert 'cache_misses' in mock_gen_scraper.performance_metrics
        assert 'dedup_time' in mock_gen_scraper.performance_metrics
        assert 'total_time' in mock_gen_scraper.performance_metrics

    def test_get_performance_metrics(self, mock_gen_scraper):
        """Test the get_performance_metrics() method."""
        metrics = mock_gen_scraper.get_performance_metrics()

        assert isinstance(metrics, dict)
        assert 'hash_computations' in metrics
        assert 'cache_hits' in metrics

    def test_get_statistics(self, mock_gen_scraper):
        """Test the get_statistics() method returns stats correctly."""
        stats = mock_gen_scraper.get_statistics()

        assert isinstance(stats, dict)
        assert 'calendar_events' in stats
        assert 'pdf_events' in stats
        assert 'duplicates_removed' in stats
        assert 'total_unique' in stats

    @pytest.mark.asyncio
    async def test_error_handling_in_parallel_pipeline(self, mock_gen_scraper):
        """Test that errors in one source don't break the entire pipeline."""
        # Make calendar extraction fail by returning an exception in gather
        def mock_extraction():
            async def _mock_cal():
                raise Exception("Calendar extraction failed")
            return _mock_cal()

        # Use AsyncMock but have it return success for other sources
        mock_gen_scraper.extract_from_calendars_async = AsyncMock()
        mock_gen_scraper.extract_from_calendars_async.side_effect = Exception("Calendar extraction failed")

        mock_gen_scraper.extract_from_pdfs_async = AsyncMock(
            return_value=pd.DataFrame(self.sample_events_pdf())
        )
        mock_gen_scraper.extract_from_websites_async = AsyncMock(
            return_value=pd.DataFrame()
        )

        # Pipeline should complete (asyncio.gather with return_exceptions=True)
        result = await mock_gen_scraper.run_pipeline_parallel()

        # Should have results or be empty - we're just testing it completes
        assert isinstance(result, pd.DataFrame)

    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self, mock_gen_scraper):
        """Test that performance metrics are collected during execution."""
        mock_gen_scraper.extract_from_calendars_async = AsyncMock(
            return_value=pd.DataFrame(self.sample_events_calendar())
        )
        mock_gen_scraper.extract_from_pdfs_async = AsyncMock(
            return_value=pd.DataFrame()
        )
        mock_gen_scraper.extract_from_websites_async = AsyncMock(
            return_value=pd.DataFrame()
        )

        result = await mock_gen_scraper.run_pipeline_parallel()

        # Check performance stats were recorded
        perf = mock_gen_scraper.stats.get('performance', {})
        assert perf.get('execution_mode') == 'parallel'
        assert 'total_duration_seconds' in perf

    def test_log_statistics_output(self, mock_gen_scraper, caplog):
        """Test that log_statistics produces appropriate output."""
        # Set up some stats
        mock_gen_scraper.stats['calendar_events'] = 10
        mock_gen_scraper.stats['pdf_events'] = 5
        mock_gen_scraper.stats['total_unique'] = 14
        mock_gen_scraper.stats['duplicates_removed'] = 1

        with caplog.at_level(logging.INFO):
            mock_gen_scraper.log_statistics()

        # Verify logging occurred (mocked logger was called)
        mock_gen_scraper.logger.info.assert_called()

    def test_source_tracking(self, mock_gen_scraper):
        """Test that event source attribution is tracked correctly."""
        events = [
            {'Name_of_the_Event': 'Event A', 'Start_Date': '2025-10-25', 'URL': 'url1'},
            {'Name_of_the_Event': 'Event B', 'Start_Date': '2025-10-26', 'URL': 'url2'},
        ]

        mock_gen_scraper.deduplicate_events(events, source='test_source')

        # Check that source is tracked
        assert len(mock_gen_scraper.extraction_source_map) == 2
        for hash_val, source in mock_gen_scraper.extraction_source_map.items():
            assert source == 'test_source'

    @pytest.mark.asyncio
    async def test_empty_extraction_handling(self, mock_gen_scraper):
        """Test handling when all sources return empty results."""
        mock_gen_scraper.extract_from_calendars_async = AsyncMock(
            return_value=pd.DataFrame()
        )
        mock_gen_scraper.extract_from_pdfs_async = AsyncMock(
            return_value=pd.DataFrame()
        )
        mock_gen_scraper.extract_from_websites_async = AsyncMock(
            return_value=pd.DataFrame()
        )

        result = await mock_gen_scraper.run_pipeline_parallel()

        assert result.empty, "Result should be empty DataFrame"
        assert mock_gen_scraper.stats['total_unique'] == 0

    def test_batch_processing_settings(self, mock_gen_scraper):
        """Test that batch processing parameters are initialized."""
        assert hasattr(mock_gen_scraper, 'batch_size')
        assert mock_gen_scraper.batch_size > 0
        assert hasattr(mock_gen_scraper, 'hash_cache_limit')
        assert mock_gen_scraper.hash_cache_limit > 0

    def test_deduplication_with_special_characters(self, mock_gen_scraper):
        """Test deduplication handles special characters correctly."""
        events = [
            {
                'Name_of_the_Event': 'Café & Bar - Salsa Night',
                'Start_Date': '2025-10-25',
                'URL': 'https://example.com/café-salsa'
            },
            {
                'Name_of_the_Event': 'Café & Bar - Salsa Night',
                'Start_Date': '2025-10-25',
                'URL': 'https://example.com/café-salsa'
            }
        ]

        unique = mock_gen_scraper.deduplicate_events(events, source='test')

        assert len(unique) == 1, "Should deduplicate despite special characters"

    def test_deduplication_with_unicode(self, mock_gen_scraper):
        """Test deduplication handles Unicode correctly."""
        events = [
            {
                'Name_of_the_Event': 'Danzas Folklóricas Latinoamericanas',
                'Start_Date': '2025-10-25',
                'URL': 'https://example.com/danzas'
            },
            {
                'Name_of_the_Event': 'Danzas Folklóricas Latinoamericanas',
                'Start_Date': '2025-10-25',
                'URL': 'https://example.com/danzas'
            }
        ]

        unique = mock_gen_scraper.deduplicate_events(events, source='test')

        assert len(unique) == 1, "Should deduplicate Unicode events"

    @pytest.mark.asyncio
    async def test_parallel_vs_sequential_same_result(self, mock_gen_scraper):
        """Test that parallel and sequential modes produce same deduplicated results."""
        calendar_data = self.sample_events_calendar()
        pdf_data = self.sample_events_pdf()

        # Configure for parallel test
        mock_gen_scraper.extract_from_calendars_async = AsyncMock(
            return_value=pd.DataFrame(calendar_data.copy())
        )
        mock_gen_scraper.extract_from_pdfs_async = AsyncMock(
            return_value=pd.DataFrame(pdf_data.copy())
        )
        mock_gen_scraper.extract_from_websites_async = AsyncMock(
            return_value=pd.DataFrame()
        )

        result_parallel = await mock_gen_scraper.run_pipeline_parallel()

        # Reset stats for sequential test
        mock_gen_scraper.seen_events.clear()
        mock_gen_scraper.extraction_source_map.clear()
        mock_gen_scraper.stats['duplicates_removed'] = 0
        mock_gen_scraper.stats['duplicates_kept'] = 0

        # Configure for sequential test
        mock_gen_scraper.extract_from_calendars_async = AsyncMock(
            return_value=pd.DataFrame(calendar_data.copy())
        )
        mock_gen_scraper.extract_from_pdfs_async = AsyncMock(
            return_value=pd.DataFrame(pdf_data.copy())
        )
        mock_gen_scraper.extract_from_websites_async = AsyncMock(
            return_value=pd.DataFrame()
        )

        result_sequential = await mock_gen_scraper.run_pipeline_sequential()

        # Results should be identical
        assert len(result_parallel) == len(result_sequential)


class TestGeneralScraperResourceManagement:
    """Tests for resource management and cleanup."""

    def test_context_manager_entry_exit(self):
        """Test context manager functionality."""
        import sys
        sys.path.insert(0, 'src')

        with patch('llm.LLMHandler') as mock_llm, \
             patch('rd_ext_v2.ReadExtractV2') as mock_read_extract, \
             patch('read_pdfs_v2.ReadPDFsV2') as mock_read_pdfs, \
             patch('gen_scraper.yaml.safe_load', return_value={}):

            from gen_scraper import GeneralScraper

            scraper = GeneralScraper()
            scraper.logger = MagicMock()
            scraper.circuit_breaker = MagicMock()

            with scraper as s:
                assert s is scraper


class TestGeneralScraperEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def mock_gen_scraper(self):
        """Create mocked scraper instance."""
        import sys
        sys.path.insert(0, 'src')

        with patch('llm.LLMHandler') as mock_llm, \
             patch('rd_ext_v2.ReadExtractV2') as mock_read_extract, \
             patch('read_pdfs_v2.ReadPDFsV2') as mock_read_pdfs, \
             patch('gen_scraper.yaml.safe_load', return_value={}):

            from gen_scraper import GeneralScraper

            scraper = GeneralScraper()
            scraper.logger = MagicMock()
            scraper.circuit_breaker = MagicMock()
            scraper.db_writer = None
            return scraper

    def test_missing_fields_in_events(self, mock_gen_scraper):
        """Test handling events with missing fields."""
        events = [
            {'Name_of_the_Event': 'Event A'},  # Missing URL and date
            {'URL': 'https://example.com'},  # Missing name and date
            {},  # Completely empty
        ]

        unique = mock_gen_scraper.deduplicate_events(events, source='test')

        # Should not crash despite missing fields
        assert isinstance(unique, list)

    def test_null_and_none_values(self, mock_gen_scraper):
        """Test handling None and null values."""
        events = [
            {'Name_of_the_Event': None, 'Start_Date': None, 'URL': None},
            {'Name_of_the_Event': 'Test', 'Start_Date': '2025-10-25', 'URL': 'https://test.com'},
        ]

        unique = mock_gen_scraper.deduplicate_events(events, source='test')

        assert isinstance(unique, list)

    def test_very_large_event_list(self, mock_gen_scraper):
        """Test handling large number of events."""
        # Create 1000 events
        events = [
            {
                'Name_of_the_Event': f'Event {i}',
                'Start_Date': '2025-10-25',
                'URL': f'https://example.com/{i}'
            }
            for i in range(1000)
        ]

        unique = mock_gen_scraper.deduplicate_events(events, source='test')

        assert len(unique) == 1000
        assert mock_gen_scraper.stats['duplicates_kept'] == 1000


# Test execution
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
