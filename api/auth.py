import os
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets

security = HTTPBasic()

def get_admin_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """
    Validate HTTP Basic credentials against environment variables:
    API_ADMIN_USER, API_ADMIN_PASS
    """
    admin_user = os.getenv("API_ADMIN_USER")
    admin_pass = os.getenv("API_ADMIN_PASS")

    if not admin_user or not admin_pass:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Admin credentials are not configured in environment."
        )

    is_valid_user = secrets.compare_digest(credentials.username, admin_user)
    is_valid_pass = secrets.compare_digest(credentials.password, admin_pass)

    if not (is_valid_user and is_valid_pass):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid authentication credentials",
                            headers={"WWW-Authenticate": "Basic"})
    return credentials.username
