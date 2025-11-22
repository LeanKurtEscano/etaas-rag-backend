from pydantic import BaseModel, EmailStr


class SellerDetails(BaseModel):
    name: str
    businessName: str
    shopName: str
    addressLocation: str
    addressOfOwner: str
    contactNumber: str
    email: EmailStr