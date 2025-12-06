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

        # 입력 이미지는 224x224 jpeg만 쓸 계획이라고 했으니까 여기도 224로 맞춰 줄게
        self.to_tensor = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),  # [0,1], (C,H,W)
            ]
        )

    def _params_for_intensity(self, intensity: str) -> dict:
        """
        Spring에서 내려오는 intensity(LOW/MEDIUM/HIGH)를
        PGD eps / inpaint strength / steps 쪽으로 매핑
        """
        # 기본값: MEDIUM
        eps = 8.0                   # PGD epsilon (pixel space)
        gen_strength = 0.6          # SD inpaint strength
        gen_guidance_scale = 3.0    # classifier-free guidance
        gen_steps = 30              # diffusion steps

        if intensity == "LOW":
            eps = 4.0
            gen_strength = 0.4
            gen_guidance_scale = 2.0
            gen_steps = 20
        elif intensity == "HIGH":
            eps = 12.0
            gen_strength = 0.8
            gen_guidance_scale = 4.0
            gen_steps = 40

        return {
            "eps": eps,
            "gen_strength": gen_strength,
            "gen_guidance_scale": gen_guidance_scale,
            "gen_steps": gen_steps,
        }

    def run_attack_raw(
        self,
        pil_image: Image.Image,
        intensity: str = "MEDIUM",
        prompt: str = "",
    ):
        """
        - pil_image: 입력 얼굴 이미지
        - intensity: LOW / MEDIUM / HIGH
        - prompt: SD prompt (지금은 ""로 들어옴)
        """
        # PIL -> tensor (1,3,H,W), [0,1]
        img_tensor = self.to_tensor(pil_image).unsqueeze(0)
        img_tensor = img_tensor.to(torch.float32)

        # intensity에 따라 attack 파라미터 세팅
        attack_params = self._params_for_intensity(intensity)

        logger.info(
            f"Running PGDAttackVAEInpaint.attack() intensity={intensity} params={attack_params}"
        )

        result = self.attacker.attack(
            original_images=img_tensor,
            prompt=prompt or "",
            **attack_params,
        )
        return result


def get_pgd_service() -> PGDAttackService:
    global _pgd_service
    if _pgd_service is None:
        _pgd_service = PGDAttackService()
    return _pgd_service