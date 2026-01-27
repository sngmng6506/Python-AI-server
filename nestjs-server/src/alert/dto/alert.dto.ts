import {
    IsBoolean,
    IsInt,
    IsISO8601,
    IsArray,
    ValidateNested,
  } from 'class-validator';
  import { Type } from 'class-transformer';
  import { AnomalousFeatureDto } from './anomaly-feature.dto';
  
  /**
   * Python AI 서버 → NestJS 이상 알람 
   */
  export class AnomalyAlertDto {
    @IsISO8601()
    timestamp: string;
  
    @IsBoolean()
    anomaly_detected: boolean;
  
    @IsInt()
    num_anomalous_features: number;
  
    @IsArray()
    @ValidateNested({ each: true })
    @Type(() => AnomalousFeatureDto)
    anomalous_features: AnomalousFeatureDto[];
  }
  