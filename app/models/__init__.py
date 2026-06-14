"""ORM models. Importing this package registers all models with the declarative
base so that `Base.metadata.create_all()` can build the schema in one shot."""

from app.models.document import Document  # noqa: F401
from app.models.role import Permission, Role, role_permissions  # noqa: F401
from app.models.user import User, user_roles  # noqa: F401
