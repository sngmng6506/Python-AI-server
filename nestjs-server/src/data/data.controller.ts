import { Controller, Post} from '@nestjs/common';
import { DataService } from './data.service';

@Controller('data')
export class DataController {

  constructor(private readonly dataService: DataService) {}

    @Post('send')
    async send() {
        const data = this.dataService.generateRandomData();
        const result = await this.dataService.sendData(data);
        return result;

    }
    }
    