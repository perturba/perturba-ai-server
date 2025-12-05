import logging
from typing import Optional, Tuple
from PIL import Image
import torch
import torchvision.transforms as T
from pgd import PGDAttackVAEInpaint

logger = logging.getLogger(__name__)

_pgd_service: Optional["PGDAttackService"] = None


#-------------------------------
# PGDAttackVAEInpaint 호출용 서비스
#--------------------------------
class PGDAttackService:

    def __init__(self):
        logger.info("Initializing PGDAttackVAEInpaint...")
        self.attacker = PGDAttackVAEInpaint(verbose=True)
        self.to_tensor = T.Compose(
            [
                T.Resize((512, 512)),
                T.ToTensor(),  # [0,1], (C,H,W)
            ]
        )

    def run_attack_raw(
        self,
        pil_image: Image.Image,
        prompt: str = "",
    ):
        #원본 result dict 자체를 반환 (job_runner에서 더 많은 정보 활용하기 위함)
        img_tensor = self.to_tensor(pil_image).unsqueeze(0)
        img_tensor = img_tensor.to(torch.float32)

        logger.info("Running PGDAttackVAEInpaint.attack()...")
        result = self.attacker.attack(
            original_images=img_tensor,
            prompt=prompt or "",
        )
        return result


def get_pgd_service() -> PGDAttackService:
    global _pgd_service
    if _pgd_service is None:
        _pgd_service = PGDAttackService()
    return _pgd_service