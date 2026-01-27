import { Module } from '@nestjs/common';
import { AlertController } from './alert.controller';
import { AlertService } from './alert.service';
import { TypeOrmModule } from '@nestjs/typeorm';
import { AlertEntity } from './alert.entity';

@Module({
  imports: [TypeOrmModule.forFeature([AlertEntity])],
  controllers: [AlertController],
  providers: [AlertService],
})
export class AlertModule {}
