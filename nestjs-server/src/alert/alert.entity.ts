import { Entity, PrimaryGeneratedColumn, Column } from 'typeorm';

@Entity()
export class AlertEntity {
  @PrimaryGeneratedColumn()
  id: number;

  @Column()
  timestamp: string;

  @Column()
  anomaly_detected: boolean;

}