"""Repository layer for data access patterns."""
from .address_repository import AddressRepository
from .url_repository import URLRepository

__all__ = ['AddressRepository', 'URLRepository']
