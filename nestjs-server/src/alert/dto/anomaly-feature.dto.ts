import { IsInt, IsNumber } from 'class-validator';

/**
 * 이상치로 판단된 feature 하나
 */
export class AnomalousFeatureDto {
  @IsInt()
  feature_idx: number;

  @IsNumber()
  score: number;
}
