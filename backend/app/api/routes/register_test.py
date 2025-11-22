from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db
from app.models.seller import Seller
from app.schemas.seller import SellerDetails

router = APIRouter()


    

@router.post("/sellers")
async def create_seller(data: SellerDetails, db: AsyncSession = Depends(get_db)):
    new_seller = Seller(
        name=data.name,
        businessName=data.businessName,
        shopName=data.shopName,
        addressLocation=data.addressLocation,
        addressOfOwner=data.addressOfOwner,
        contactNumber=data.contactNumber,
        email=data.email,
    )

    db.add(new_seller)
    await db.commit()
    await db.refresh(new_seller)

    return new_seller