import { Injectable, Logger } from '@nestjs/common';
import { AnomalyAlertDto } from './dto/alert.dto';
import { Repository } from 'typeorm';
import { AlertEntity } from './alert.entity';
import { InjectRepository } from '@nestjs/typeorm';


@Injectable()
export class AlertService {
  private readonly logger = new Logger(AlertService.name);

  constructor(
    @InjectRepository(AlertEntity)
    private alertRepository: Repository<AlertEntity>,
  ){}

  async logAnomalyAlert(dto: AnomalyAlertDto) {
    this.logger.log(
      `Anomaly detected at ${dto.timestamp}`
    );
  }

  async sendAnomalyAlert(dto: AnomalyAlertDto) {
    try{
        const alert = this.alertRepository.create({
            timestamp: dto.timestamp,
            anomaly_detected: dto.anomaly_detected,
        });

        const result = await this.alertRepository.save(alert);

        this.logger.log(`Alert logged successfully: ${result.id}`);
        return {
            status: 'ok',
        };

    } catch (error) {
        this.logger.error('Error logging anomaly alert', error);
        throw error;
    }
    }
}
