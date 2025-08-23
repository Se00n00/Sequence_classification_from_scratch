import { Component, signal, inject, ViewChild, ElementRef } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { InputTypeService } from '../input-type.service';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Chart, LinearScale, CategoryScale, PointElement, LineElement, LineController, Filler } from 'chart.js';

Chart.register(LinearScale, CategoryScale, PointElement, LineElement, LineController, Filler);
import * as XLSX from 'xlsx';
import { RouterLink } from '@angular/router';
import { forkJoin } from 'rxjs';


@Component({
  selector: 'app-home',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterLink],
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css']
})
export class HomeComponent {
  @ViewChild('densityPlotCanvas') densityPlotCanvas!: ElementRef<HTMLCanvasElement>;
  public Math = Math;
  constructor(public inputTypeService: InputTypeService, private http: HttpClient) {}

  text: string = '';
  sentiment: string = '';
  chart: any;
  isButtonPressed = signal(false);
  apiResponse = signal<any>(null);
  softmaxResponse = signal<number[]>([]);
  batchSentimentResponse = signal<any[]>([]);
  inputText = '';
  feedbackMessage = signal<string | null>(null);
  sheets = signal<string[]>([]);
  selectedSheet = signal<string>('');
  fileColumns = signal<string[]>([]);
  fileData = signal<any[]>([]);
  workbook: XLSX.WorkBook | null = null;
  selectedColumnIndex = signal<number>(0);
  selectedColumnData = signal<any[]>([]);

  onDragOver(event: any) {
    event.preventDefault();
    event.stopPropagation();
  }

  onDragLeave(event: any) {
    event.preventDefault();
    event.stopPropagation();
  }

  onFileDrop(event: any) {
    event.preventDefault();
    event.stopPropagation();
    const files = event.dataTransfer.files;
    if (files.length > 0) {
      this.handleFile(files[0]);
    }
  }

  toggleInputType() {
    this.inputTypeService.activeInputType.set(this.inputTypeService.activeInputType() === 'text' ? 'file' : 'text');
  }

  submitText() {
    this.analyze(this.inputText);
  }

  onFileSelected(event: any) {
    const file = event.target.files[0];
    this.handleFile(file);
  }

  handleFile(file: File) {
    const reader = new FileReader();
    reader.onload = (e: any) => {
      const data = new Uint8Array(e.target.result);
      this.workbook = XLSX.read(data, { type: 'array' });
      this.sheets.set(this.workbook.SheetNames);
      if (this.sheets().length > 0) {
        this.selectSheet(this.sheets()[0]);
      }
    };
    reader.readAsArrayBuffer(file);
  }

  selectSheet(sheet: string) {
    this.selectedSheet.set(sheet);
    this.onSheetChange();
  }

  handleKeyDown(event: KeyboardEvent) {
    const currentIndex = this.sheets().indexOf(this.selectedSheet());
    if (event.key === 'ArrowDown') {
      const nextIndex = (currentIndex + 1) % this.sheets().length;
      this.selectSheet(this.sheets()[nextIndex]);
    } else if (event.key === 'ArrowUp') {
      const prevIndex = (currentIndex - 1 + this.sheets().length) % this.sheets().length;
      this.selectSheet(this.sheets()[prevIndex]);
    } else if (event.key === 'ArrowRight') {
      this.nextColumn();
    } else if (event.key === 'ArrowLeft') {
      this.previousColumn();
    }
  }

  selectColumn(index: number) {
    this.selectedColumnIndex.set(index);
    this.updateSelectedColumnData();
  }

  nextColumn() {
    this.selectedColumnIndex.set((this.selectedColumnIndex() + 1) % this.fileColumns().length);
    this.updateSelectedColumnData();
  }

  previousColumn() {
    this.selectedColumnIndex.set((this.selectedColumnIndex() - 1 + this.fileColumns().length) % this.fileColumns().length);
    this.updateSelectedColumnData();
  }

  onSheetChange() {
    if (this.workbook) {
      const worksheet = this.workbook.Sheets[this.selectedSheet()];
      const fileData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
      this.fileData.set(fileData);
      if (fileData.length > 0) {
        this.fileColumns.set(fileData[0] as string[]);
        this.selectedColumnIndex.set(0);
        this.updateSelectedColumnData();
      }
      // In a real application, you would process the file data here.
      // For now, we will just log it to the console.
      console.log(this.fileData());
    }
  }

  updateSelectedColumnData() {
    const columnIndex = this.selectedColumnIndex();
    const columnData = this.fileData().map(row => row[columnIndex]);
    this.selectedColumnData.set(columnData.slice(1)); // Get all data excluding header
  }

  analyzeBatchSentiment() {
    const texts = this.selectedColumnData().filter(data => data !== null && data !== undefined && data !== '');
    if (texts.length === 0) {
      this.feedbackMessage.set('No data in selected column to analyze.');
      return;
    }

    this.feedbackMessage.set('Analyzing sentiment...');
    this.batchSentimentResponse.set([]); // Clear previous results

    const requests = texts.map(text =>
      this.http.post<any>('/predict', { review: String(text) }) // Send one by one
    );

    forkJoin(requests).subscribe({
      next: (responses) => {
        const allProcessedResults: any[] = [];
        responses.forEach(response => {
          let item: any;
          // Try to find the sentiment scores in different possible locations
          if (response && response.predictions && response.predictions.length > 0) {
            item = response.predictions[0];
          } else if (response && (response.Negative !== undefined || response.Positive !== undefined)) {
            item = response; // Response itself contains the scores
          } else if (response && response.prediction && (response.prediction.Negative !== undefined || response.prediction.prediction.Positive !== undefined)) {
            item = response.prediction; // Scores wrapped in 'prediction'
          } else if (response && response.result && (response.result.Negative !== undefined || response.result.Positive !== undefined)) {
            item = response.result; // Scores wrapped in 'result'
          }

          if (item && (item.Negative !== undefined || item.Positive !== undefined)) {
            if (item.Negative > item.Positive) {
              allProcessedResults.push({ sentiment: 'Negative', score: item.Negative });
            } else {
              allProcessedResults.push({ sentiment: 'Positive', score: item.Positive });
            }
          } else {
            console.warn('Unexpected response format for sentiment analysis:', response);
          }
        });
        console.log('Batch sentiment analysis complete. Results:', allProcessedResults);
        this.batchSentimentResponse.set(allProcessedResults);
        this.feedbackMessage.set('Batch sentiment analysis complete!');
      },
      error: (error) => {
        console.error('Error analyzing batch sentiment:', error);
        this.feedbackMessage.set('Error analyzing batch sentiment. Please try again.');
      }
    });
  }

  analyze(text: string) {
    if (!text) {
      this.feedbackMessage.set('Please enter text to analyze.');
      return;
    }

    this.http.post<any>('/predict', { review: String(text) }).subscribe({
      next: (response) => {
        // Assuming response is { Negative: score, Positive: score } or { predictions: [{ Negative: score, Positive: score }] }
        let item: any;
        if (response.predictions && response.predictions.length > 0) {
          item = response.predictions[0];
        } else {
          item = response; // Assume response itself is the item
        }

        if (item.Negative > item.Positive) {
          this.sentiment = 'Negative';
        } else {
          this.sentiment = 'Positive';
        }
        this.apiResponse.set(item); // Store the raw response
        this.softmaxResponse.set([item.Negative, item.Positive]); // Store softmax scores

        // Update chart based on sentiment
        const score = item.Positive - item.Negative; // Simple score for chart
        this.updateChart(score);
      },
      error: (error) => {
        console.error('Error analyzing text sentiment:', error);
        this.feedbackMessage.set('Error analyzing text sentiment. Please try again.');
      }
    });
  }

  generateGaussianData(mean: number, stdDev: number, count: number) {
    const data = [];
    for (let i = 0; i < count; i++) {
      const x = -5 + (10 * i / count);
      const y = Math.exp(-0.5 * Math.pow((x - mean) / stdDev, 2)) / (stdDev * Math.sqrt(2 * Math.PI));
      data.push({ x, y });
    }
    return data;
  }

  updateChart(score: number) {
    if (this.chart) {
      this.chart.destroy();
    }

    const positiveMean = 2;
    const negativeMean = -2;
    const stdDev = 1;
    const dataPointCount = 1000;

    const positiveData = this.generateGaussianData(positiveMean, stdDev, dataPointCount);
    const negativeData = this.generateGaussianData(negativeMean, stdDev, dataPointCount);

    if (score > 0) {
      positiveData.forEach(p => p.y *= (1 + score/5));
    } else if (score < 0) {
      negativeData.forEach(p => p.y *= (1 - score/5));
    }


    const ctx = this.densityPlotCanvas?.nativeElement;
    if (!ctx) {
      console.warn('Canvas element not found or not ready. Chart cannot be created.');
      return;
    }
    this.chart = new Chart(ctx, {
      type: 'line',
      data: {
        datasets: [{
          label: 'Positive',
          data: positiveData,
          borderColor: 'rgba(75, 192, 192, 1)',
          backgroundColor: 'rgba(75, 192, 192, 0.5)',
          fill: true,
          tension: 0.4,
          pointRadius: 0,
          borderWidth: 1
        }, {
          label: 'Negative',
          data: negativeData,
          borderColor: 'rgba(255, 99, 132, 1)',
          backgroundColor: 'rgba(255, 99, 132, 0.5)',
          fill: true,
          tension: 0.1,
          pointRadius: 0,
          borderWidth: 1
        }]
      },
      options: {
        scales: {
          x: {
            type: 'linear',
            position: 'bottom',
            grid: {
              display: false
            },
            ticks: {
              display: false
            },
            border: {
              display: false
            }
          },
          y: {
            grid: {
              display: false
            },
            ticks: {
              display: false
            },
            border: {
              display: false
            }
          }
        },
        plugins: {
          legend: {
            display: false
          }
        }
      }}
    );
  }
  resetSignals(){
    this.sheets.set([]);
    this.selectedSheet.set('');
    this.fileColumns.set([]);
    this.fileData.set([]);
    this.workbook = null;
    this.selectedColumnIndex.set(0);
    this.selectedColumnData.set([]);
    this.batchSentimentResponse.set([]);
    this.feedbackMessage.set(null);
  }
}