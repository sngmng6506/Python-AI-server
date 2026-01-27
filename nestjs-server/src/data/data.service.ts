import { Injectable, Logger } from '@nestjs/common';
import axios, { AxiosInstance } from 'axios';
import { SendDataDto } from './dto/data.dto';
import { gzip } from 'zlib';
import { promisify } from 'util';
import axiosRetry from 'axios-retry';

const gzipAsync = promisify(gzip);

@Injectable()
export class DataService {
  private readonly axiosInstance: AxiosInstance;
  private readonly logger = new Logger(DataService.name);

  constructor() {

    // Axios 인스턴스 생성
    this.axiosInstance = axios.create({
      baseURL: 'http://localhost:9000',
      timeout: 50000,
    });
    
    const MAX_RETRIES = 5;
    const getDelay = (retryCount: number) => {
        const base = 1000; // 1초
        return base * Math.pow(2, retryCount - 1); //1초부터 시작해서 2배씩 증가
      };

    // Retry 설정 - 
    axiosRetry(this.axiosInstance, {
        retries: MAX_RETRIES,
        retryDelay: (retryCount) => {
            const delay = getDelay(retryCount);
            return delay;
        },
        retryCondition: (error) => {
        if (!error.response) return true; // 네트워크 에러
        return error.response.status >= 500; // 5xx 서버 에러
        
        },
        onRetry: (retryCount, error) => {
            const delay = getDelay(retryCount);
            this.logger.warn(`Retrying request 
                (${retryCount}/(${MAX_RETRIES}) after ${delay/1000}sec) : ${error.message}`);

        },
    });
    }


  async sendData(dto: SendDataDto) {
    try {
      // 전송 데이터 크기 계산 (JSON 직렬화 후)
      const jsonString = JSON.stringify(dto);
      const sizeInBytes = Buffer.byteLength(jsonString, 'utf-8');
      const sizeInKB = (sizeInBytes / 1024).toFixed(2);
      this.logger.log(`Sending data size: ${sizeInKB} KB`);

      // 압축
      const compressed = await gzipAsync(jsonString);
      const compressedSizeInKB = (compressed.length / 1024).toFixed(2);
      this.logger.log(`Compressed data size: ${compressedSizeInKB} KB`);

      const response = await this.axiosInstance.post('/data-enqueue', compressed);
      return response.data;


    } catch (error) {
      this.logger.error('Error sending data to Python server:', error);
      throw error;
    }
  }


  generateRandomData(): SendDataDto {
    // 25,000개의 key-value 쌍 생성
    const data: Record<string, number> = {};
    for (let i = 0; i < 25000; i++) {
      data[`feature_${i}`] = Math.random();
    }

    return { data };
  }
}  


