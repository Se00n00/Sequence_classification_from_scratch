import { Injectable, signal } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class InputTypeService {
  public activeInputType = signal('text');
}
