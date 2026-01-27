import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { DataModule } from './data/data.module';
import { AlertModule } from './alert/alert.module';

@Module({
  imports: [
    TypeOrmModule.forRoot({
      type: 'postgres',
      host: 'localhost',
      port: 5433,
      username: 'postgres',
      password: '1234',
      database: 'nestjs-server',
      autoLoadEntities: true,
      synchronize: true, // ⚠️ 개발용 only
      connectTimeoutMS: 30000,
      retryAttempts: 3,
      retryDelay: 1000,
      ssl: false,
      logging: false,
    }),
    DataModule,
    AlertModule,
  ],
})
export class AppModule {}
