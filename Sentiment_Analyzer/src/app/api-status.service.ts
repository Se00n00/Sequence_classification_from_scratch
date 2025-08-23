import { Injectable, signal } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class ApiStatusService {
  isApiLive = signal<boolean>(false);

  constructor() { }
}