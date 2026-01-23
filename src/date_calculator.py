"""
Date Calculator for LLM Function Calling

Provides reliable date calculations for temporal queries like 'next month',
'next weekend', 'this week', etc. Used by the LLM via function calling to
ensure accurate date range calculations.

Author: Claude Code
Version: 1.1.0
"""

from datetime import datetime, timedelta
import logging
import os
import yaml

logger = logging.getLogger(__name__)

# Load config for temporal expressions
_config = None

def _load_config():
    """Load configuration from config.yaml."""
    global _config
    if _config is None:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        with open(config_path, 'r') as f:
            _config = yaml.safe_load(f)
    return _config


def calculate_date_range(temporal_phrase: str, current_date: str) -> dict:
    """
    Calculate start and end dates for temporal expressions.

    This function implements the algorithmic approach for date calculations,
    ensuring consistent and accurate results for all temporal queries.

    Args:
        temporal_phrase (str): Temporal expression from user query
            Examples: 'next week', 'next month', 'this weekend', 'tomorrow'
        current_date (str): Current date in YYYY-MM-DD format

    Returns:
        dict: Date range with keys:
            - start_date (str): Start date in YYYY-MM-DD format
            - end_date (str): End date in YYYY-MM-DD format
            - month_number (int, optional): Month number for month queries
            - year (int, optional): Year for month/year queries
            - time_filter (str, optional): Time filter like '18:00:00' for 'tonight'
            - dow_filter (list, optional): Day of week filter [5,6,0] for weekends
    """
    try:
        current = datetime.strptime(current_date, "%Y-%m-%d")
    except ValueError as e:
        logger.error(f"Invalid date format: {current_date}, error: {e}")
        raise ValueError(f"current_date must be in YYYY-MM-DD format, got: {current_date}")

    temporal_phrase = temporal_phrase.lower().strip()

    # Today
    if temporal_phrase == "today":
        return {
            "start_date": current.strftime("%Y-%m-%d"),
            "end_date": current.strftime("%Y-%m-%d")
        }

    # Tonight (today with time filter)
    elif temporal_phrase == "tonight":
        return {
            "start_date": current.strftime("%Y-%m-%d"),
            "end_date": current.strftime("%Y-%m-%d"),
            "time_filter": "18:00:00"
        }

    # Tomorrow
    elif temporal_phrase == "tomorrow":
        tomorrow = current + timedelta(days=1)
        return {
            "start_date": tomorrow.strftime("%Y-%m-%d"),
            "end_date": tomorrow.strftime("%Y-%m-%d")
        }

    # Tomorrow night
    elif temporal_phrase == "tomorrow night":
        tomorrow = current + timedelta(days=1)
        return {
            "start_date": tomorrow.strftime("%Y-%m-%d"),
            "end_date": tomorrow.strftime("%Y-%m-%d"),
            "time_filter": "18:00:00"
        }

    # Yesterday
    elif temporal_phrase == "yesterday":
        yesterday = current - timedelta(days=1)
        return {
            "start_date": yesterday.strftime("%Y-%m-%d"),
            "end_date": yesterday.strftime("%Y-%m-%d")
        }

    # This week (Monday-Sunday of current calendar week)
    elif temporal_phrase == "this week":
        dow = current.weekday()  # 0=Monday, 6=Sunday

        # Calculate Monday of this week
        days_since_monday = dow  # If Monday, 0; if Tuesday, 1; etc.
        monday = current - timedelta(days=days_since_monday)

        # Calculate Sunday of this week
        days_to_sunday = 6 - dow  # If Monday, 6; if Sunday, 0
        sunday = current + timedelta(days=days_to_sunday)

        return {
            "start_date": monday.strftime("%Y-%m-%d"),
            "end_date": sunday.strftime("%Y-%m-%d")
        }

    # Next week (next Monday through next Sunday)
    elif temporal_phrase == "next week":
        dow = current.weekday()  # 0=Monday, 6=Sunday
        # Days to next Monday
        days_to_monday = (7 - dow) % 7
        if days_to_monday == 0:
            days_to_monday = 7  # Not today, next week

        next_monday = current + timedelta(days=days_to_monday)
        next_sunday = next_monday + timedelta(days=6)

        return {
            "start_date": next_monday.strftime("%Y-%m-%d"),
            "end_date": next_sunday.strftime("%Y-%m-%d")
        }

    # This weekend (Friday-Sunday of current week)
    elif temporal_phrase in ["this weekend", "coming weekend"]:
        dow = current.weekday()  # 0=Monday, 6=Sunday

        # If Monday-Thursday: upcoming Fri-Sun
        if dow in [0, 1, 2, 3]:
            days_to_friday = 4 - dow
            friday = current + timedelta(days=days_to_friday)
            sunday = friday + timedelta(days=2)
        # If Friday-Sunday: this Fri-Sun (or remaining days)
        else:
            # Calculate back to Friday of this week
            days_since_friday = dow - 4  # Friday is day 4
            friday = current - timedelta(days=days_since_friday)
            sunday = friday + timedelta(days=2)

        return {
            "start_date": friday.strftime("%Y-%m-%d"),
            "end_date": sunday.strftime("%Y-%m-%d"),
            "dow_filter": [5, 6, 0]  # Friday, Saturday, Sunday
        }

    # Next weekend (Friday-Sunday of next week)
    elif temporal_phrase == "next weekend":
        dow = current.weekday()  # 0=Monday, 6=Sunday

        # Find next Monday
        days_to_monday = (7 - dow) % 7
        if days_to_monday == 0:
            days_to_monday = 7  # Not today, next week

        next_monday = current + timedelta(days=days_to_monday)
        friday = next_monday + timedelta(days=4)  # Monday + 4 = Friday
        sunday = next_monday + timedelta(days=6)  # Monday + 6 = Sunday

        return {
            "start_date": friday.strftime("%Y-%m-%d"),
            "end_date": sunday.strftime("%Y-%m-%d"),
            "dow_filter": [5, 6, 0]  # Friday, Saturday, Sunday
        }

    # This month
    elif temporal_phrase == "this month":
        # First day of current month
        first_day = current.replace(day=1)

        # Last day of current month
        if current.month == 12:
            last_day = current.replace(day=31)
        else:
            next_month_first = current.replace(month=current.month + 1, day=1)
            last_day = next_month_first - timedelta(days=1)

        return {
            "start_date": first_day.strftime("%Y-%m-%d"),
            "end_date": last_day.strftime("%Y-%m-%d"),
            "month_number": current.month,
            "year": current.year
        }

    # Next month
    elif temporal_phrase == "next month":
        # Algorithm: month + 1, wrap if > 12, increment year if wrapped
        month = current.month
        year = current.year

        next_month = month + 1 if month < 12 else 1
        next_year = year + 1 if month == 12 else year

        # First day of next month
        first_day = datetime(next_year, next_month, 1)

        # Last day of next month
        if next_month == 12:
            last_day = datetime(next_year, 12, 31)
        else:
            following_month_first = datetime(next_year, next_month + 1, 1)
            last_day = following_month_first - timedelta(days=1)

        return {
            "start_date": first_day.strftime("%Y-%m-%d"),
            "end_date": last_day.strftime("%Y-%m-%d"),
            "month_number": next_month,
            "year": next_year
        }

    # Last month
    elif temporal_phrase == "last month":
        month = current.month
        year = current.year

        last_month = month - 1 if month > 1 else 12
        last_year = year - 1 if month == 1 else year

        # First day of last month
        first_day = datetime(last_year, last_month, 1)

        # Last day of last month
        if last_month == 12:
            last_day = datetime(last_year, 12, 31)
        else:
            current_month_first = datetime(last_year, last_month + 1, 1)
            last_day = current_month_first - timedelta(days=1)

        return {
            "start_date": first_day.strftime("%Y-%m-%d"),
            "end_date": last_day.strftime("%Y-%m-%d"),
            "month_number": last_month,
            "year": last_year
        }

    # This year
    elif temporal_phrase == "this year":
        year = current.year
        return {
            "start_date": f"{year}-01-01",
            "end_date": f"{year}-12-31",
            "year": year
        }

    # Next year
    elif temporal_phrase == "next year":
        year = current.year + 1
        return {
            "start_date": f"{year}-01-01",
            "end_date": f"{year}-12-31",
            "year": year
        }

    # Last year
    elif temporal_phrase == "last year":
        year = current.year - 1
        return {
            "start_date": f"{year}-01-01",
            "end_date": f"{year}-12-31",
            "year": year
        }

    # Load config for day names and time periods
    config = _load_config()
    day_names = config.get('temporal_expressions', {}).get('day_names', {})
    time_periods = config.get('temporal_expressions', {}).get('time_periods', {})

    # Day-specific queries (e.g., "Monday", "Tuesday", "Friday")
    for day_name, target_dow in day_names.items():
        if temporal_phrase == day_name:
            current_dow = current.weekday()

            # If today is the target day, return today
            if current_dow == target_dow:
                return {
                    "start_date": current.strftime("%Y-%m-%d"),
                    "end_date": current.strftime("%Y-%m-%d"),
                    "dow_filter": [target_dow]
                }

            # Otherwise, find the next occurrence of this day
            days_ahead = (target_dow - current_dow) % 7
            if days_ahead == 0:
                days_ahead = 7  # Next week if today is the target day

            target_date = current + timedelta(days=days_ahead)
            return {
                "start_date": target_date.strftime("%Y-%m-%d"),
                "end_date": target_date.strftime("%Y-%m-%d"),
                "dow_filter": [target_dow]
            }

    # Last [day] queries (e.g., "last monday", "last wednesday")
    for day_name, target_dow in day_names.items():
        if temporal_phrase == f"last {day_name}":
            current_dow = current.weekday()

            # Calculate days back to the last occurrence of this day
            if current_dow == target_dow:
                # If today is the target day, go back 7 days to last week
                days_back = 7
            else:
                # Otherwise calculate days back
                days_back = (current_dow - target_dow) % 7
                if days_back == 0:
                    days_back = 7  # Go to last week, not today

            target_date = current - timedelta(days=days_back)
            return {
                "start_date": target_date.strftime("%Y-%m-%d"),
                "end_date": target_date.strftime("%Y-%m-%d"),
                "dow_filter": [target_dow]
            }

    # Day + time period queries (e.g., "Monday morning", "Friday night", "Saturday evening")
    for day_name, target_dow in day_names.items():
        for period_name, period_config in time_periods.items():
            if temporal_phrase == f"{day_name} {period_name}":
                current_dow = current.weekday()

                # If today is the target day, return today with time filter
                if current_dow == target_dow:
                    result = {
                        "start_date": current.strftime("%Y-%m-%d"),
                        "end_date": current.strftime("%Y-%m-%d"),
                        "dow_filter": [target_dow]
                    }

                    # Add time filters
                    if period_config.get('start_time'):
                        result['time_filter'] = period_config['start_time']
                    if period_config.get('end_time'):
                        result['end_time_filter'] = period_config['end_time']

                    return result

                # Otherwise, find the next occurrence of this day
                days_ahead = (target_dow - current_dow) % 7
                if days_ahead == 0:
                    days_ahead = 7  # Next week if today is the target day

                target_date = current + timedelta(days=days_ahead)
                result = {
                    "start_date": target_date.strftime("%Y-%m-%d"),
                    "end_date": target_date.strftime("%Y-%m-%d"),
                    "dow_filter": [target_dow]
                }

                # Add time filters
                if period_config.get('start_time'):
                    result['time_filter'] = period_config['start_time']
                if period_config.get('end_time'):
                    result['end_time_filter'] = period_config['end_time']

                return result

    # Unsupported temporal phrase
    else:
        logger.warning(f"Unsupported temporal phrase: {temporal_phrase}")
        raise ValueError(f"Unsupported temporal phrase: {temporal_phrase}")


def _generate_temporal_phrase_enum():
    """Generate the enum list of supported temporal phrases based on config."""
    # Base temporal phrases
    base_phrases = [
        "today",
        "tonight",
        "tomorrow",
        "tomorrow night",
        "yesterday",
        "this week",
        "next week",
        "this weekend",
        "coming weekend",
        "next weekend",
        "this month",
        "next month",
        "last month",
        "this year",
        "next year",
        "last year"
    ]

    # Load config to get day names and time periods
    config = _load_config()
    day_names = list(config.get('temporal_expressions', {}).get('day_names', {}).keys())
    time_periods = list(config.get('temporal_expressions', {}).get('time_periods', {}).keys())

    # Add day-specific phrases
    all_phrases = base_phrases + day_names

    # Add "last [day]" phrases (e.g., "last monday", "last wednesday")
    for day in day_names:
        all_phrases.append(f"last {day}")

    # Add day + time period combinations
    for day in day_names:
        for period in time_periods:
            all_phrases.append(f"{day} {period}")

    return sorted(all_phrases)


# Function calling schema for LLM
CALCULATE_DATE_RANGE_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate_date_range",
        "description": """Calculate start and end dates for temporal expressions in user queries.
        Use this function whenever the user mentions a time period like 'next week', 'next month',
        'this weekend', etc. This ensures accurate date calculations for SQL queries.""",
        "parameters": {
            "type": "object",
            "properties": {
                "temporal_phrase": {
                    "type": "string",
                    "description": "The temporal expression from the user query",
                    "enum": _generate_temporal_phrase_enum()
                },
                "current_date": {
                    "type": "string",
                    "description": "Current date in YYYY-MM-DD format (provided in context)",
                    "pattern": "^\\d{4}-\\d{2}-\\d{2}$"
                }
            },
            "required": ["temporal_phrase", "current_date"]
        }
    }
}
