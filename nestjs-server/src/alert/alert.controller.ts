import { Controller, Post, Body } from '@nestjs/common';
import { AlertService } from './alert.service';
import { AnomalyAlertDto } from './dto/alert.dto';



@Controller('api/v1')
export class AlertController {
  constructor(private readonly alertService: AlertService) {}

  @Post('alert')
  async anomalyAlert(@Body() dto: AnomalyAlertDto) {
    await this.alertService.logAnomalyAlert(dto);
    await this.alertService.sendAnomalyAlert(dto);

    return {
      status: 'ok',
    };
  }
}
