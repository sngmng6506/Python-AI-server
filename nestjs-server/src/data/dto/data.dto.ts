import { IsObject } from 'class-validator';


export class SendDataDto {
  @IsObject()
  data: Record<string, number>; // key-value 쌍 
}