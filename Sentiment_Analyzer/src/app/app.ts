import { Component, signal, OnInit, OnDestroy } from '@angular/core';
import { InputTypeService } from './input-type.service';
import { RouterOutlet } from '@angular/router';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { interval, Subscription } from 'rxjs';
import { switchMap, catchError } from 'rxjs/operators';
import { of } from 'rxjs';
import { ApiStatusService } from './api-status.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, HttpClientModule, CommonModule],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App implements OnInit, OnDestroy {
  protected readonly title = signal('Sentiment_Analyzer');
  public status = signal('waking up');
  private statusSubscription!: Subscription;

  constructor(private http: HttpClient, public apiStatusService: ApiStatusService, public inputTypeService: InputTypeService) {}

  ngOnInit() {
    this.statusSubscription = interval(5000) // Check every 5 seconds
      .pipe(
        switchMap(() => this.http.get("/api/", { observe: 'response', responseType: 'text' })
          .pipe(
            catchError(() => of(null))
          )
        )
      )
      .subscribe(response => {
        console.log(response)
        if (response && response.status === 200) {
          this.status.set('live');
          this.apiStatusService.isApiLive.set(true);
          this.statusSubscription.unsubscribe(); // Stop checking once live
        }
      });
  }

  ngOnDestroy() {
    if (this.statusSubscription) {
      this.statusSubscription.unsubscribe();
    }
  }
}
